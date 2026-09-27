import torch
import numpy as np

# * * * *  * * * *  * * * *   *       *   
# *        *     *  *     *   * *   * *   
# *   * *  * * *    * * * *   *   *   *   
# *     *  *     *  *     *   *       *   
# * * * *  *     *  *     *   *       *   

# THIS IS THE CORE PY CODE OF GRAM FRAMEWORK

def volume_computation2(language, video):
    """
    Computes the area (2D volume) for each pair of samples between language (shape [batch_size1, feature_dim])
    and video (shape [batch_size2, feature_dim]) using the determinant of a 2x2 matrix.

    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the area for each pair.
    """

    batch_size1 = language.shape[0]
    batch_size2 = video.shape[0]

    # Step 1: Compute difference vectors
    bary = language.unsqueeze(1)

    b = bary.expand(-1, batch_size2, -1)
    v = bary - video.unsqueeze(0)  # [B1, B2, D]

    # Step 2: Stack into a [B1, B2, 2, D] tensor
    diffs = torch.stack([b, v], dim=2)  # [B1, B2, 2, D]

    # Step 3: Compute Gram matrices G = D @ D^T for each batch pair
    G_new = torch.matmul(diffs, diffs.transpose(-1, -2))  # [B1, B2, 2, 2]

    # Compute determinant
    gram_res_det = torch.det(G_new.float())  # [B1, B2]

    # Square root of absolute determinant gives the 2D "volume" (area)
    res_new = torch.sqrt(torch.abs(gram_res_det))

    return res_new


def volume_computation3(language, video, audio):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    bary = language.unsqueeze(1)

    b = bary.expand(-1,batch_size2,-1)
    v = bary - video.unsqueeze(0)     # [B1, B2, D]
    a = bary - audio.unsqueeze(0)     # [B1, B2, D]

    # Step 2: Stack into a [B1, B2, 3, D] tensor
    diffs = torch.stack([b, v, a], dim=2)  # [B1, B2, 3, D]

    # Step 3: Compute Gram matrices G = D @ D^T for each batch pair
    G_new = torch.matmul(diffs, diffs.transpose(-1, -2))  # [B1, B2, 3, 3]

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_res_det = torch.det(G_new.float())
    # Compute the square root of the absolute value of the determinants
    res_new =  torch.sqrt(torch.abs(gram_res_det))
    #print(res.shape)
    return res_new


def volume_computation4(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4 matrix

    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles
    # Step 1: Compute difference vectors
    bary = language.unsqueeze(1)

    b = bary.expand(-1,batch_size2,-1)
    v = bary - video.unsqueeze(0)     # [B1, B2, D]
    a = bary - audio.unsqueeze(0)     # [B1, B2, D]
    s = bary - subtitles.unsqueeze(0) # [B1, B2, D]

    # Step 2: Stack into a [B1, B2, 3, D] tensor
    diffs = torch.stack([b, v, a, s], dim=2)  # [B1, B2, 3, D]

    # Step 3: Compute Gram matrices G = D @ D^T for each batch pair
    G_new = torch.matmul(diffs, diffs.transpose(-1, -2))  # [B1, B2, 3, 3]

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_res_det = torch.det(G_new.float())
    # Compute the square root of the absolute value of the determinants
    res_new =  torch.sqrt(torch.abs(gram_res_det))
    #print(res.shape)
    return res_new 


def volume_computation5(language, video, audio, subtitles, depth):
    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles, depth (shape [batch_size2, feature_dim]) using the determinant of a 5x5 matrix.

    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    - depth (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing depth features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles, depth

    # Step 1: Compute difference vectors
    bary = language.unsqueeze(1)

    b = bary.expand(-1, batch_size2, -1)
    v = bary - video.unsqueeze(0)       # [B1, B2, D]
    a = bary - audio.unsqueeze(0)       # [B1, B2, D]
    s = bary - subtitles.unsqueeze(0)   # [B1, B2, D]
    d = bary - depth.unsqueeze(0)       # [B1, B2, D]

    # Step 2: Stack into a [B1, B2, 5, D] tensor
    diffs = torch.stack([b, v, a, s, d], dim=2)  # [B1, B2, 5, D]

    # Step 3: Compute Gram matrices G = D @ D^T for each batch pair
    G_new = torch.matmul(diffs, diffs.transpose(-1, -2))  # [B1, B2, 5, 5]

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_res_det = torch.det(G_new.float())

    # Compute the square root of the absolute value of the determinants
    res_new = torch.sqrt(torch.abs(gram_res_det))

    return res_new

