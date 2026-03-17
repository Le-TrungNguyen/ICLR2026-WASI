import torch
import numpy as np
import time

def Gram_Schmidt(matrix):
    new_matrix = matrix.clone()

    original_type = new_matrix.dtype #torch.linalg.qr doesn't support helf precision types such as torch.bfloat16
    new_matrix, _ = torch.linalg.qr(new_matrix.to(dtype=torch.float32))
    new_matrix = new_matrix.to(dtype=original_type)

    return new_matrix

def set_random(shape, random_seed=233, device='cuda'):
    torch.manual_seed(np.random.RandomState(random_seed).randint(1_000_000_000))
    random_tensor = torch.randn(shape, device=device)
    return random_tensor

def decompose_tensor(tensor, previous_q=None, reuse_q=False, rank=1, device='cuda'):
    """Keep subspace"""
    n, m = tensor.shape

    rank = min(m, n, rank)

    # prepare q
    if reuse_q:
        q = previous_q.clone().detach()
    else:
        q = set_random((m, rank), device=device)

    # Tính p
    p = torch.matmul(tensor, q)#, out=p)


    # Chuẩn hóa p
    # p = orthogonalize(p)
    p = Gram_Schmidt(p)

    # Tính q
    q = torch.matmul(tensor.t(), p)#, out=q)
    return p, q

def decompose_tensor_keep_projection(tensor, previous_p=None, reuse_p=False, rank=1, device='cuda', orthogonalization_time=None, matmuls_time=None):
    n, m = tensor.shape

    rank = min(m, n, rank)

    # prepare q
    if reuse_p:
        p = previous_p.clone().detach()
    else:
        q = set_random((m, rank), device=device)

        # Tính p
        p = torch.matmul(tensor, q)#, out=p)


        # Chuẩn hóa p
        # p = orthogonalize(p)
        p = Gram_Schmidt(p)
    
    if matmuls_time is not None: start_matmul = time.time()
    # Tính q
    q = torch.matmul(tensor.t(), p)#, out=q)
    p = torch.matmul(tensor, q)#, out=p)

    if matmuls_time is not None: matmuls_time[-1] += time.time() - start_matmul

    if orthogonalization_time is not None: start_orthogonalization = time.time()
    p = Gram_Schmidt(p)
    if orthogonalization_time is not None: orthogonalization_time[-1] += time.time() - start_orthogonalization
    return p, q