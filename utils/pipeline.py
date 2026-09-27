import os
import torch 
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm 
# from apex import amp
from collections import defaultdict
from torch.nn.utils import clip_grad_norm_
from evaluation import evaluation_registry
from .save import ModelSaver
from .tool import NoOp
from .logger import LOGGER, RunningMeter
from .sched import get_lr_sched
from torch.cuda.amp import autocast, GradScaler
import wandb


def train(model, optimizer, optimizer_pot, train_loader, val_loaders, args, start_step=0, verbose_time=False):
    run_cfg = args.run_cfg
    dataset_cfg = args.data_cfg.train[0]
    if dist.get_rank() == 0:
        pbar = tqdm(total=run_cfg.num_train_steps, initial=start_step)
        model_saver = ModelSaver(os.path.join(run_cfg.output_dir, 'ckpt'),remove_before_ckpt=run_cfg.remove_before_ckpt)
    else:
        pbar = NoOp()
        model_saver = NoOp()
        
    loss_moving_averagetors ={}
    metric_logger_dict = defaultdict(dict)
    global_step = start_step

    scaler = GradScaler()

    best_indicator = {}
    evaluate_fn = evaluation_registry[model.config.evaluation_type]

    for step, (name, batch) in enumerate(train_loader):
        ndata = train_loader.ndata
        task = name.split('--')[0]

        if run_cfg.fp16:
            with autocast():
                # 计算损失字典，分开计算模型损失和判别器损失
                loss_dict = model(batch, task=task, compute_loss=True)
                
                # 主模型损失
                loss_main = sum([v for k, v in loss_dict.items() if 'loss_pot' not in k])  # 排除掉loss_pot
                # 判别器损失
                loss_pot = loss_dict.get('loss_pot', torch.zeros((), device=loss_main.device, dtype=loss_main.dtype, requires_grad=True))
                # 汇总总损失
                loss = loss_main + loss_pot
                loss_dict['total_loss'] = loss

                # 记录损失到wandb
                loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                if dist.get_rank() == 0:
                    wandb.log(loss_dict)
                    
        else:
            # 计算损失字典，分开计算模型损失和判别器损失
            loss_dict = model(batch, task=task, compute_loss=True)
            
            # 主模型损失
            loss_main = sum([v for k, v in loss_dict.items() if 'loss_pot' not in k])  # 排除掉loss_pot
            # 判别器损失
            loss_pot = loss_dict.get('loss_pot', torch.zeros((), device=loss_main.device, dtype=loss_main.dtype, requires_grad=True))

            # 汇总总损失
            loss = loss_main + loss_pot
            loss_dict['total_loss'] = loss

            # 记录损失到wandb
            loss_dict = {k: v.item() for k, v in loss_dict.items()}
            if dist.get_rank() == 0:
                wandb.log(loss_dict)

        # 初始化loss moving averages            
        if name not in loss_moving_averagetors:
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()

        # 累积损失
        for k, v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)

        global_step += 1
        # 学习率调度
        lr_ratio = get_lr_sched(global_step, run_cfg)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['init_lr'] * lr_ratio

        if global_step %  50  == 0:
            LOGGER.info({name: averagetor.val for name, averagetor in loss_moving_averagetors.items()})

        # 更新主模型和判别器参数

        if run_cfg.fp16:
            # 反向传播主模型损失
            optimizer.zero_grad()
            scaler.scale(loss_main).backward()

            # 反向传播判别器损失
            optimizer_pot.zero_grad()
            scaler.scale(loss_pot).backward()

        else:
            # 反向传播主模型损失
            optimizer.zero_grad()
            loss_main.backward()

            # 反向传播判别器损失
            optimizer_pot.zero_grad()
            loss_pot.backward()

        # 如果不使用DDP，进行梯度同步
        if not run_cfg.use_ddp:
            works = []
            for p in model.parameters():
                if p.grad is not None:
                    works.append(dist.all_reduce(p.grad, async_op=True))
            for work in works:          
                work.wait()

        # 更新主模型参数
        if run_cfg.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
            optimizer.zero_grad()

        # 更新判别器参数
        if run_cfg.fp16:
            scaler.step(optimizer_pot)
            scaler.update()
        else:
            optimizer_pot.step()

        pbar.update(1)


        
        if (global_step+1) % run_cfg.valid_steps == 0:
            eval_log = evaluate_fn(model, val_loaders, run_cfg, global_step)

            if dist.get_rank() == 0:
                for task_name, val_log in eval_log.items():
                    for eval_name, metric in val_log.items():
                        eval_name = task_name +'_' +eval_name 
                        metric_logger_dict[eval_name][str(global_step)] = metric
                        LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--===========\n")
                        LOGGER.info(metric)
                        best_name = get_best_name(eval_name, metric)
                        if best_name is not None:
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric[best_name] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric[best_name]
                                best_indicator[eval_name] = True 
                            else:
                                best_indicator[eval_name] = False 
                            best_step = metric_logger_dict[eval_name]['best_step']
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}=======\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])          
                
                model_saver.save(model, global_step, optimizer,best_indicator, run_cfg.save_best)
    

        if global_step >= run_cfg.num_train_steps:
            break
    pbar.close()






    
  
def test(model, test_loader, run_cfg):
    
    evaluate_fn = evaluation_registry[model.config.evaluation_type]
    eval_log = evaluate_fn(model, test_loader, run_cfg, global_step=0)
    if dist.get_rank()==0:  
        for task_name, val_log in eval_log.items():
            for eval_name, metric in val_log.items():
                eval_name = task_name +'_' +eval_name 
                # TB_LOGGER.log_scaler_dict({f"eval/{eval_name}/test_{k}": v
                #                 for k, v in metric.items() if not isinstance(v,str)})
                LOGGER.info(f"==== evaluation--{eval_name}========\n")
                LOGGER.info(metric)




def get_best_name(eval_name, metric):
    if eval_name.startswith('cap'):
        return 'CIDEr'
    elif eval_name.startswith('qa'):
        return 'accuracy'
    elif eval_name.startswith('ret'):
        if 'video_r1' in metric:
            return 'video_r1'
        if 'volume_T2D_r1' in metric:
            return 'volume_T2D_r1'
        if 'gramian_value' in metric:
            return 'gramian_value'
    elif eval_name.startswith('pt'):
        return None 
    else:
        raise NotImplementedError

