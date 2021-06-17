import argparse
import sys
import time
from scipy import stats

import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from PyHessian.pyhessian import hessian  # Hessian computation
from models import GeneralModel
from models.statistics import Metrics
from models.statistics.Flops import FLOPCounter
from models.statistics.Saliency import Saliency
from utils.model_utils import find_right_model, linear_CKA, kernel_CKA, batch_CKA, cka, cka_batch
from utils.system_utils import *
from torch.optim.lr_scheduler import StepLR
from PyHessian.density_plot import get_esd_plot  # ESD plot
from PyHessian.pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, \
    orthnormal, set_grad
# from rigl_torch.RigL import RigLScheduler
import gc
from nngeometry import metrics
from nngeometry.object import pspace
from utils.fim import fim_diag, unit_trace_diag
from nonparstat.Cucconi import cucconi_test


class DefaultTrainer:
    """
    Implements generalised computer vision classification with pruning
    """

    def __init__(self,
                 model: GeneralModel,
                 loss: GeneralModel,
                 optimizer: Optimizer,
                 device,
                 arguments: argparse.Namespace,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 metrics: Metrics,
                 criterion: GeneralModel,
                 scheduler: StepLR,
                 # pruner: RigLScheduler
                 ):

        self._test_loader = test_loader
        self._train_loader = train_loader
        self._test_model = None
        self._fim_loader = None
        self.gradient_adtest = []
        self.loss_test = []
        self._stable = False
        self._overlap_queue = []
        self._loss_function = loss
        self._model = model
        self._arguments = arguments
        self._optimizer = optimizer
        self._device = device
        self._global_steps = 0
        self.out = metrics.log_line
        self.patience = 0
        DATA_MANAGER.set_date_stamp(addition=arguments.run_name)
        self._writer = SummaryWriter(os.path.join(DATA_MANAGER.directory, RESULTS_DIR, DATA_MANAGER.stamp, SUMMARY_DIR))
        self._metrics: Metrics = metrics
        self._metrics.init_training(self._writer)
        self._acc_buffer = []
        self._loss_buffer = []
        self._elapsed_buffer = []
        self._criterion = criterion
        self._scheduler = scheduler
        # self._pruner = pruner
        self.ts = None
        self.old_score = None
        self.old_grads = None
        self.gradient_flow = 0
        self._variance = 0
        self.mask1 = self._model.mask.copy()
        self.mask2 = None
        self.newgrad = None
        self.newweight = None
        self.scores = None
        self.count = 0
        self._step = 0.97
        self._percentage = 0.999
        batch = next(iter(self._test_loader))
        self.saliency = Saliency(model, device, batch[0][:8])
        self._metrics.write_arguments(arguments)
        self._flopcounter = FLOPCounter(model, batch[0][:8], self._arguments.batch_size, device=device)
        self._metrics.model_to_tensorboard(model, timestep=-1)

    def _batch_iteration(self,
                         x: torch.Tensor,
                         y: torch.Tensor,
                         train: bool = True):
        """ one iteration of forward-backward """

        # unpack
        x, y = x.to(self._device).float(), y.to(self._device)

        # update metrics
        self._metrics.update_batch(train)

        # record time
        if "cuda" in str(self._device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        # forward pass
        accuracy, loss, out = self._forward_pass(x, y, train=train)
        # backward pass
        # breakpoint()
        if train:
            self._backward_pass(loss)

        # record time
        if "cuda" in str(self._device):
            end.record()
            torch.cuda.synchronize(self._device)
            time = start.elapsed_time(end)
        else:
            time = 0

        # free memory
        for tens in [out, y, x, loss]:
            tens.detach()

        return accuracy, loss.item(), time

    def _forward_pass(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      train: bool = True):
        """ implementation of a forward pass """

        if train:
            self._optimizer.zero_grad()
            if self._model.is_maskable:
                self._model.apply_weight_mask()

        out = self._model(x).squeeze()
        loss = self._loss_function(
            output=out,
            target=y,
            weight_generator=self._model.parameters(),
            model=self._model,
            criterion=self._criterion
        )
        accuracy = self._get_accuracy(out, y)
        return accuracy, loss, out

    def _backward_pass(self, loss):
        """ implementation of a backward pass """

        loss.backward()
        self._model.insert_noise_for_gradient(self._arguments.grad_noise)
        if self._arguments.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._arguments.grad_clip)
        # if self._arguments.prune_criterion == "RigL" and self._pruner():
        self._optimizer.step()
        if self._model.is_maskable:
            self._model.apply_weight_mask()

    def smooth(self, scalars, weight):  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed

    def ntk(self, model, inp):
        """Calculate the neural tangent kernel of the model on the inputs.
        Returns the gradient feature map along with the tangent kernel.
        """
        out = model(inp.to(self._device).float())
        p_vec = torch.nn.utils.parameters_to_vector(model.parameters())
        p, = p_vec.shape
        n, outdim = out.shape
        # assert outdim == 1, "cant handle output dim higher than 1 for now"

        # this is the transpose jacobian (grad y(w))^T)
        features = torch.zeros(n, p, requires_grad=False)

        for i in range(outdim):  # for loop over data points
            model.zero_grad()
            out[0][i].backward(retain_graph=True)
        p_grad = torch.tensor([], requires_grad=False).to(self._device)
        for p in model.parameters():
            p_grad = torch.cat((p_grad, p.grad.reshape(-1)))
        features[0, :] = p_grad

        tk = features @ features.t()  # compute the tangent kernel
        return features, tk

    def _epoch_iteration(self):
        """ implementation of an epoch """
        self._model.train()
        self.out("\n")

        self._acc_buffer, self._loss_buffer = self._metrics.update_epoch()
        overlap = 0
        proj_norm = 0
        mean_abs_mag_grad = 0
        count = 0
        gradient_norm = []
        gradient_adtest = []
        loss_test = []
        for batch_num, batch in enumerate(self._train_loader):
            self.out(f"\rTraining... {batch_num}/{len(self._train_loader)}", end='')

            if self._model.is_tracking_weights:
                self._model.save_prev_weights()

            # Perform one batch iteration
            acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)
            # self._metrics.add(torch.norm(self.ntk(self._model, batch[0])[1], p=1), key="ntk/iter")
            if False:
                # # hessian stuff
                self._optimizer.zero_grad()
                hessian_comp = hessian(self._model, self._loss_function,
                                       data=batch, cuda=True)
                # EXACT
                # proj_norm += hessian_comp.norm_top_gradient(dim=10).detach().cpu() / group_product(hessian_comp.gradsH,
                #                                                                                    hessian_comp.gradsH).detach().cpu()
                # OVERLAP
                # Hg = hessian_vector_product(hessian_comp.gradsH, hessian_comp.params, hessian_comp.gradsH)
                # # gTHg = group_product(Hg, hessian_comp.gradsH).detach().cpu()
                # running_overlap = group_product(normalization(Hg), normalization(hessian_comp.gradsH)).detach().cpu()
                # overlap += running_overlap

                count += 1
                # New Grad Format
                newgrad = []
                for grad in hessian_comp.gradsH:
                    if len(grad.shape) != 1:
                        newgrad.append(grad)
                i = 0
                for grad, mask in zip(newgrad, self._model.mask.values()):
                    newgrad[i] = grad * mask
                    i += 1

                # New Weight Format
                newweight = []
                for param in hessian_comp.params:
                    if len(param.shape) != 1:
                        newweight.append(param)
                i = 0
                for param, mask in zip(newweight, self._model.mask.values()):
                    newweight[i] = param * mask
                    i += 1
                for i in range(len(newweight)):
                    if len(newweight[i].shape) == 4:
                        newweight[i] = torch.mean(newweight[i], (2, 3)).T
                    else:
                        newweight[i] = newweight[i].T

                gradient_norm.append(
                    sum(torch.norm(t, p=1).detach().cpu() for t in newgrad) / sum([len(k) for k in newgrad]))

                gradient_adtest.append(
                    sum(torch.norm(t, p=2).detach().cpu() for t in newgrad) / sum([len(k) for k in newgrad]))
                self._metrics.add(
                    sum(torch.norm(t, p=1).detach().cpu() for t in newgrad) / sum([len(k) for k in newgrad]),
                    key="criterion/gradientflow")
                self._metrics.add(torch.norm(torch.cat([torch.flatten(x.detach().cpu()) for x in newgrad]), p=1) / sum(
                    [len(k) for k in newgrad]),
                                  key="criterion/gradientflow")
                self._metrics.add(
                    sum([torch.norm(param, p=1).detach().cpu() for param in newweight]) / sum(
                        [len(param) for param in newweight]),
                    key="criterion/weigthmagnitude")
                self._optimizer.zero_grad()
                # out = self._model(batch[0].to(self._device).float())
                # params, grads = get_params_grad(self._model)
                # gradients = torch.autograd.grad(out, params, grad_outputs=torch.ones_like(out))
                # breakpoint()
                self._model.train()

            if self._model.is_tracking_weights:
                self._model.update_tracked_weights(self._metrics.batch_train)

            self._acc_buffer.append(acc)
            self._loss_buffer.append(loss)

            loss_test.append(loss)
            self._metrics.add(loss, key="loss/step")

            self._elapsed_buffer.append(elapsed)

            self._log(batch_num)

            self._check_exit_conditions_epoch_iteration()
            self._scheduler.step()
        self._model.eval()
        # features, tk = self.ntk(self._model, torch.unsqueeze(batch[0][0], 0))
        # if self.mask2 is not None:
        #     self._metrics.add(torch.norm(tk-self.tk)/torch.norm(self.tk), key='criterion/ntk')
        # self.tk = tk
        # TODO: Add max number of itertations
        # if overlap / count > 0.85:
        #     self._stable = True
        # self._metrics.add(overlap / count, key="criterion/batchoverlap")
        if False:
            self._metrics.add(sum(gradient_norm) / count, key="criterion/efficient_gradient_flow")
            self._metrics.add(np.var(gradient_norm), key="criterion/efficient_gradient_flow_variance")
            from statsmodels.tsa.stattools import adfuller
            self._metrics.add(
                adfuller(loss_test)[1].astype(np.float32),
                key="loss/pvalue")
            if adfuller(loss_test)[1].astype(np.float32) < 0.05:
                self.patience += 1
                if self.patience == 3:
                    self._stable = True
            else:
                self.patience=0

            self._metrics.add(
                adfuller(gradient_adtest)[1].astype(np.float32),
                key="loss/pvalue_GF")

            if len(self.gradient_adtest) != 0:

                # if stats.kstest(self.gradient_adtest, gradient_adtest, mode='exact')[1] * 100 > 10:
                #     self._stable = True
                self._metrics.add(
                    stats.kstest(self.gradient_adtest, gradient_adtest, mode='exact')[1].astype(np.float32),
                    key="criterion/pvalue")

                # print("HIIIIIIIIIII")
                # print(stats.kstest(self.gradient_adtest, gradient_adtest, mode='exact'))
                self._metrics.add(cucconi_test(np.asarray(self.gradient_adtest), np.asarray(gradient_adtest))[1],
                                  key="criterion/cucconipvalue")
                self._metrics.add(stats.kstest(self.gradient_adtest, gradient_adtest, mode='exact')[0],
                                  key="criterion/kstest")

            self.gradient_adtest = gradient_adtest.copy()
            self.loss_test = loss_test.copy()
            if self._variance != 0:
                self._metrics.add(np.abs(np.var(gradient_norm) - self._variance),
                                  key="criterion/efficient_gradient_flow_variance_dist")
            self._variance = np.var(gradient_norm)

            if self.gradient_flow != 0:
                self._metrics.add(torch.abs(sum(gradient_norm) / count - self.gradient_flow),
                                  key="criterion/gradient_flow_dist")
            self.gradient_flow = sum(gradient_norm) / count
            # # self._metrics.add(proj_norm / count, key="criterion/proj_norm")
            # # self._stable = True
            #
            # # GET SCORE MASK DISTANCE
            self.mask1 = self._model.mask.copy()
            if self._stable == False and "Structured" not in self._arguments.prune_criterion:
                # steps = self._criterion.steps.copy()
                all_scores, grads_abs, log10, norm_factor = self._criterion.get_weight_saliencies(self._train_loader)
                # all_scores *= torch.cat([torch.flatten(x) for _, x in self.mask1.items()])
                # create mask
                num_params_to_keep = int(len(all_scores) * (1 - 0.98))
                if num_params_to_keep < 1:
                    num_params_to_keep += 1
                elif num_params_to_keep > len(all_scores):
                    num_params_to_keep = len(all_scores)

                # threshold
                threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
                # breakpoint()
                acceptable_score = threshold[-1]

                # all_scores = all_scores/torch.norm(all_scores)

                # prune
                for name, grad in grads_abs.items():
                    self.mask1[name] = ((grad / norm_factor) > acceptable_score).__and__(
                        self.mask1[name].bool()).float().to(self._device)

                if self.mask2 is not None:
                    if False:
                        import copy
                        self._test_model = copy.deepcopy(self._model)
                        self._test_model.mask = self.mask1
                        self._test_model.apply_weight_mask()
                        self._test_model.add_hooks()
                        for batch_num, batch in enumerate(self._train_loader):
                            self._test_model(batch[0].to(self._device))
                            break
                        activations1 = []
                        for value in self._test_model.hooks.values():
                            activations1.append(value)

                        # Weight mask 2

                        self._test_model = copy.deepcopy(self._model)
                        self._test_model.mask = self.mask2
                        self._test_model.apply_weight_mask()
                        self._test_model.add_hooks()
                        for batch_num, batch in enumerate(self._train_loader):
                            self._test_model(batch[0].to(self._device))
                            break
                        activations2 = []
                        for value in self._test_model.hooks.values():
                            activations2.append(value)
                        cka_distances = np.zeros(len(activations1))
                        import math
                        for j in range(len(activations1)):
                            cka_distances[j] = cka_batch(activations1[j], activations1[j])
                        for l, cka in enumerate(cka_distances):
                            self._metrics.add(cka, key="cka/layer" + str(l))
                        self._metrics.add(np.mean(cka_distances), key="criterion/cka")
                    # breakpoint()
                    maskdist = (sum([torch.dist(mask1, mask2, p=0) for mask1, mask2 in
                                     zip(self.mask1.values(), self.mask2.values())]) / 2) / sum(
                        [len(torch.nonzero(t)) for t in self.mask1.values()])
                    # breakpoint()
                    # maskdist = 1 - (sum([torch.dist(mask1, mask2, p=0) for mask1, mask2 in
                    #                      zip(self.mask1.values(), self.mask2.values())]) / 2) / sum(
                    #     [len(torch.nonzero(t)) for t in self.mask1.values()])

                    # i = 0
                    # for param, mask in zip(newweight, self.mask1.values()):
                    #     newweight[i] = param * mask
                    #     i += 1
                    self._metrics.add(maskdist, key="criterion/maskdist")
                    self._metrics.add(torch.norm(all_scores, p=1) / len(all_scores), key="criterion/all_scores")
                    # cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                    self._metrics.add(torch.dist(self.scores, all_scores, p=1) / len(all_scores),
                                      key="criterion/scoredist")
                    # if maskdist > 50 and self.count > 4:
                    # self._stable = True
                    # self._criterion.steps = [self._criterion.limit]
                    # Prune
                    # percentage = self._step*maskdist
                    # breakpoint()
                    # self._criterion.prune(percentage=percentage, train_loader=self._train_loader, manager=DATA_MANAGER)
                    # self._model.mask = self.mask1.copy()
                    # self._model.apply_weight_mask()
                    # print("final percentage after snip:", self._model.pruned_percentage)
                    # Pop first step
                    # self._criterion.steps.pop(0)
                    # self.mask2 = None
                    # if len(self._criterion.steps) == 0:
                    #     self._stable = True
                    # self.count = 0
                    # return
                else:
                    self._metrics.add(torch.tensor(0), key="criterion/maskdist")
                # breakpoint()
                self.mask2 = self.mask1.copy()
                # self.newgrad = newgrad.copy() * self.mask2
                self.newweight = newweight.copy()
                self.scores = all_scores
                self.count += 1

    def _log(self, batch_num: int):
        """ logs to terminal and tensorboard if the time is right"""

        if (batch_num % self._arguments.eval_freq) == 0:
            # validate on test and train set
            train_acc, train_loss = np.mean(self._acc_buffer), np.mean(self._loss_buffer)
            test_acc, test_loss, test_elapsed = self.validate()
            self._elapsed_buffer += test_elapsed

            # log metrics
            self._add_metrics(test_acc, test_loss, train_acc, train_loss)

            # reset for next log
            self._acc_buffer, self._loss_buffer, self._elapsed_buffer = [], [], []

            # print to terminal
            self.out(self._metrics.printable_last)

    def validate(self):
        """ validates the model on test set """

        self.out("\n")

        # init test mode
        self._model.eval()
        cum_acc, cum_loss, cum_elapsed = [], [], []
        # F = metrics.FIM_MonteCarlo(model=self._model, loader=self._fim_loader, representation=pspace.PMatDiag,
        #                            device='cuda')
        # self._metrics.add(F.trace(), key="criterion/trFIM")
        with torch.no_grad():
            for batch_num, batch in enumerate(self._test_loader):
                acc, loss, elapsed = self._batch_iteration(*batch, self._model.training)
                cum_acc.append(acc)
                cum_loss.append(loss),
                cum_elapsed.append(elapsed)
                self.out(f"\rEvaluating... {batch_num}/{len(self._test_loader)}", end='')
        self.out("\n")

        # put back into train mode
        self._model.train()

        return float(np.mean(cum_acc)), float(np.mean(cum_loss)), cum_elapsed

    def _add_metrics(self, test_acc, test_loss, train_acc, train_loss):
        """
        save metrics
        """

        sparsity = self._model.pruned_percentage
        spasity_index = 2 * ((sparsity * test_acc) / (1e-8 + sparsity + test_acc))

        flops_per_sample, total_seen = self._flopcounter.count_flops(self._metrics.batch_train)

        self._metrics.add(train_acc, key="acc/train")
        self._metrics.add(train_loss, key="loss/train")
        self._metrics.add(test_loss, key="loss/test")
        self._metrics.add(test_acc, key="acc/test")
        self._metrics.add(sparsity, key="sparse/weight")
        self._metrics.add(self._model.structural_sparsity, key="sparse/node")
        self._metrics.add(spasity_index, key="sparse/hm")
        self._metrics.add(np.log(self._model.compressed_size), key="sparse/log_disk_size")
        self._metrics.add(np.mean(self._elapsed_buffer), key="time/gpu_time")
        self._metrics.add(int(flops_per_sample), key="time/flops_per_sample")
        self._metrics.add(np.log10(total_seen), key="time/flops_log_cum")
        if torch.cuda.is_available():
            self._metrics.add(torch.cuda.memory_allocated(0), key="cuda/ram_footprint")
        self._metrics.timeit()

    def train(self):
        """ main training function """

        # setup data output directories:
        setup_directories()
        save_codebase_of_run(self._arguments)
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "calling_command.txt"), str(" ".join(sys.argv)))

        # data gathering
        self._fim_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self._train_loader.dataset, [i for i in range(50)]))
        # self._fim_loader = self._train_loader
        epoch = self._metrics._epoch

        self._model.train()
        if self._arguments.structured_prior == 1:
            # get structured criterion
            from models.criterions.StructuredEFGit import StructuredEFGit
            criterion = StructuredEFGit(limit=self._arguments.pruning_limit-0.2, model=self._model)
            criterion.prune(train_loader=self._train_loader, manager=DATA_MANAGER)
            self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
                                               params=self._model.parameters(),
                                               lr=self._arguments.learning_rate,
                                               weight_decay=self._arguments.l2_reg)
            self._metrics.model_to_tensorboard(self._model, timestep=epoch)

        try:

            self.out(
                f"{PRINTCOLOR_BOLD}Started training{PRINTCOLOR_END}"
            )

            # if self._arguments.skip_first_plot:
            #     self._metrics.handle_weight_plotting(0, trainer_ns=self)
            if "Early" in self._arguments.prune_criterion:
                # for i in range(10):
                # self._metrics.handle_weight_plotting(epoch, trainer_ns=self)
                grow_prune = True
                while self._stable == False:
                    self.out("Network has not reached stable state")
                    self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")
                    # do epoch
                    self._epoch_iteration()

                    # Test Growing
                    # if epoch == self._arguments.prune_to:
                    # self._criterion.prune(self._arguments.pruning_limit,
                    #                       train_loader=self._train_loader,
                    #                       manager=DATA_MANAGER)
                    # self._criterion.steps.append(self._arguments.pruning_limit)
                    # Grow
                    # if epoch > self._arguments.prune_to and epoch%3==0:
                    #     self._criterion.grow(0.1, self._train_loader)
                    #     self._criterion.prune(self._arguments.pruning_limit,
                    #                           train_loader=self._train_loader,
                    #                           manager=DATA_MANAGER)
                    #     self._criterion.steps.append(self._arguments.pruning_limit)
                    if epoch == self._arguments.prune_to:
                        self._stable = True
                    epoch += 1
                    # self._metrics.handle_weight_plotting(epoch, trainer_ns=self)
            # else:
            #     self._stable = True
            # if snip we prune before training
            if self._arguments.prune_criterion in SINGLE_SHOT:
                self._criterion.prune(self._arguments.pruning_limit,
                                      train_loader=self._train_loader,
                                      manager=DATA_MANAGER)
                if self._arguments.prune_criterion in STRUCTURED_SINGLE_SHOT:
                    self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
                                                       params=self._model.parameters(),
                                                       lr=self._arguments.learning_rate,
                                                       weight_decay=self._arguments.l2_reg)
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)
            # Reset optimizer
            # self._criterion.cut_lonely_connections()
            # self._optimizer = find_right_model(OPTIMS, self._arguments.optimizer,
            #                                    params=self._model.parameters(),
            #                                    lr=self._arguments.learning_rate,
            #                                    weight_decay=self._arguments.l2_reg)
            # do training
            for epoch in range(epoch, self._arguments.epochs + epoch):
                self.out(f"\n\n{PRINTCOLOR_BOLD}EPOCH {epoch} {PRINTCOLOR_END} \n\n")

                # do epoch
                self._epoch_iteration()
                # plotting
                # if (epoch % self._arguments.plot_weights_freq) == 0 and self._arguments.plot_weights_freq > 0:
                #     self._metrics.handle_weight_plotting(epoch, trainer_ns=self)
                # do all related to pruning
                self._handle_pruning(epoch)

                # save what needs to be saved
                self._handle_backing_up(epoch)

            if self._arguments.skip_first_plot:
                self._metrics.handle_weight_plotting(epoch + 1, trainer_ns=self)

            # example last save
            save_models([self._model, self._metrics], "finished")

        except KeyboardInterrupt as e:
            self.out(f"Killed by user: {e} at {time.time()}")
            save_models([self._model, self._metrics], f"KILLED_at_epoch_{epoch}")
            sys.stdout.flush()
            DATA_MANAGER.write_to_file(
                os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
            self._writer.close()
            exit(69)
        except Exception as e:
            self._writer.close()
            report_error(e, self._model, epoch, self._metrics)

        # flush prints
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"), self._metrics.log)
        self._writer.close()

    def _handle_backing_up(self, epoch):
        if (epoch % self._arguments.save_freq) == 0 and epoch > 0:
            self.out("\nSAVING...\n")
            save_models(
                [self._model, self._metrics],
                f"save_at_epoch_{epoch}"
            )
        sys.stdout.flush()
        DATA_MANAGER.write_to_file(
            os.path.join(RESULTS_DIR, DATA_MANAGER.stamp, OUTPUT_DIR, "log.txt"),
            self._metrics.log
        )

    def _handle_pruning(self, epoch):
        if self._is_pruning_time(epoch):
            if self._is_not_finished_pruning():
                self.out("\nPRUNING...\n")
                # Here we call SNIP-it
                self._criterion.prune(
                    percentage=self._arguments.pruning_rate,
                    train_loader=self._train_loader,
                    manager=DATA_MANAGER
                )
                if self._arguments.prune_criterion in DURING_TRAINING:
                    self._optimizer = find_right_model(
                        OPTIMS, self._arguments.optimizer,
                        params=self._model.parameters(),
                        lr=self._arguments.learning_rate,
                        weight_decay=self._arguments.l2_reg
                    )
                    self._metrics.model_to_tensorboard(self._model, timestep=epoch)
                if self._model.is_rewindable:
                    self.out("rewinding weights to checkpoint...\n")
                    self._model.do_rewind()
            if self._model.is_growable:
                self.out("growing too...\n")
                self._criterion.grow(self._arguments.growing_rate)

        if self._is_checkpoint_time(epoch):
            self.out(f"\nCreating weights checkpoint at epoch {epoch}\n")
            self._model.save_rewind_weights()

    def _is_not_finished_pruning(self):
        return self._arguments.pruning_limit > self._model.pruned_percentage \
               or \
               (
                       self._arguments.prune_criterion in DURING_TRAINING
                       and
                       self._arguments.pruning_limit > self._model.structural_sparsity
               )

    @staticmethod
    def _get_accuracy(output, y):
        # predictions = torch.round(output)
        predictions = output.argmax(dim=-1, keepdim=True).view_as(y)
        correct = y.eq(predictions).sum().item()
        return correct / output.shape[0]

    def _is_checkpoint_time(self, epoch: int):
        return epoch == self._arguments.rewind_to and self._model.is_rewindable

    def _is_pruning_time(self, epoch: int):
        if self._arguments.prune_criterion == "EmptyCrit":
            return False
        # Ma bet ballech abel ma ye2ta3 prune_freq epochs
        epoch -= self._arguments.prune_delay
        return (epoch % self._arguments.prune_freq) == 0 and \
               epoch >= 0 and \
               self._model.is_maskable and \
               self._arguments.prune_criterion not in SINGLE_SHOT

    def _check_exit_conditions_epoch_iteration(self, patience=1):

        time_passed = datetime.now() - DATA_MANAGER.actual_date
        # check if runtime is expired
        if (time_passed.total_seconds() > (self._arguments.max_training_minutes * 60)) \
                and \
                self._arguments.max_training_minutes > 0:
            raise KeyboardInterrupt(
                f"Process killed because {self._arguments.max_training_minutes} minutes passed "
                f"since {DATA_MANAGER.actual_date}. Time now is {datetime.now()}")
        if patience == 0:
            raise NotImplementedError("feature to implement",
                                      KeyboardInterrupt("Process killed because patience is zero"))
