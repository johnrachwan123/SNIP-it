import math
import torch
from torch.optim.optimizer import Optimizer
from PyHessian.pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, \
    orthnormal, add, group_multiply, multiply, group_subtract, neg, sub, divide
import numpy as np


class SHD(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0, hessian_power=1, single_gpu=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power)
        self.single_gpu = single_gpu
        super(SHD, self).__init__(params, defaults)

    def eigenvalues(self, params, grads, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to("cuda") for p in params
                 ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                Hv = hessian_vector_product(grads, grads, v)
                tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors

    def eigenvector_gradient_product(self, grads, eigenvectors):
        res = []
        for i in range(len(eigenvectors)):
            res.append(group_product(grads, eigenvectors[i]))
        return res

    def eigenvector_woodbury_product(self, eigenvectors, woodbury):
        res = []
        for i in range(len(eigenvectors)):
            res.append(multiply(eigenvectors[i], woodbury[i]))
        if len(res) > 1:
            result = group_add(res[0], res[1])
        else:
            result = res[0]
        for j in range(2, len(res)):
            result = group_add(result, res[j])
        return result

    def woodbury_inverse(self, grads, eigenvalues, eigenvectors, gamma=1e-3):
        # if gamma < 1e-4:
        #     gamma_damping = 1e-3
        # else:
        #     gamma_damping = gamma
        # # TODO: Implement gamma damping
        # gamma_damping = 1
        # # Absolute value of eigenvalues
        # breakpoint()

        Lmbda_abs = np.abs(eigenvalues)
        gamma_damping = max(0.9 * Lmbda_abs[-1], 1e-3)
        D_denominator = Lmbda_abs + gamma_damping * np.ones_like(Lmbda_abs)
        D = np.divide(np.ones_like(Lmbda_abs), D_denominator)
        D = np.divide(D, gamma * np.ones_like(D))
        D = torch.abs(torch.tensor(D))
        # breakpoint()
        UTg = self.eigenvector_gradient_product(grads, eigenvectors)
        DUTg = group_multiply(UTg, D)
        UDUTg = self.eigenvector_woodbury_product(eigenvectors, DUTg)
        minus_p = divide(sub(grads, UDUTg), gamma_damping)
        return minus_p

    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                                   '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                                   '\t\t\t  set to True.')

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

        # get the Hessian diagonal

        hut_traces = self.get_trace(params, grads)
        # print(sum(torch.sum(x) for x in hut_traces))
        # breakpoint()
        for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
            exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, hut_trace, value=1 - beta2)

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # make the square root, and the Hessian power
            k = group['hessian_power']
            denom = (
                    (exp_hessian_diag_sq.sqrt() ** k) /
                    math.sqrt(bias_correction2) ** k).add_(
                group['eps'])

            # make update
            p.data = p.data - \
                     group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)

        return loss
