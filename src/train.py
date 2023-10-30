import os
from utils.make_loss import SWDLoss
import numpy as np
import torch
import random
import argparse
import yaml
import pickle
from utils.load_data import get_train_dataloader, get_validation_data
import models
from utils.make_optimizer import make_optimizer
from utils.make_logger import make_logger, make_output_dir
from torchinfo import summary
from utils.kinematics import ExpandFreeQuantities, distribution_transform_fun
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from utils.template_plot import ratio_plot, plot_line
import matplotlib.pyplot as plt
import torch.multiprocessing
from scipy.stats import kstest
from tqdm import trange
import scienceplots

torch.multiprocessing.set_sharing_strategy('file_system')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True


def set_default_dtype(dtype_name):
    if dtype_name == 'float16':
        torch.set_default_dtype(torch.float16)
    elif dtype_name == 'float32':
        torch.set_default_dtype(torch.float32)
    if dtype_name == 'float64':
        torch.set_default_dtype(torch.float64)


def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f.read())
    plt.style.use(config['plot_style'])
    if 'cut_index' not in config['model']:
        config['model']['cut_index'] = None
    if config['train']['device_id'] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = config['train']['device_id']
    setup_seed(config['seed'])
    set_default_dtype(config['data']['dtype'])
    output_dir = make_output_dir(config['output_dir'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger, train_logger, val_logger, test_logger = make_logger(output_dir)
    config['data']['batch_size'] *= (torch.cuda.device_count())
    config['data']['epoch_iter_num'] = int(config['data']['epoch_iter_num'] /
                                           torch.cuda.device_count())
    logger.info('config:{:}'.format(config))
    train_loader = get_train_dataloader(
        path=os.path.join(config['data']['datasets_path'],
                          config['data']['train_data_name']),
        epoch_iter_num=config['data']['epoch_iter_num'],
        batch_event_num=config['data']['batch_event_num'],
        decay_num=config['train']['decay_num'],
        latent_dim=config['data']['latent_dim'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        dtype=config['data']['dtype'],
        train_size=config['data']['train_size'],
        distribution_transform=config['data']['distribution_transform'])
    val_expand_data, val_free_data, val_original_data = get_validation_data(
        path=os.path.join(config['data']['datasets_path'],
                          config['data']['val_data_name']),
        decay_num=config['train']['decay_num'],
        dtype=config['data']['dtype'],
        distribution_transform=config['data']['distribution_transform'])
    val_logger.info('val data size is {}'.format(val_expand_data.shape[0]))
    logger.info('load data successfully')
    criterion = SWDLoss(
        gamma=config['train']['gamma'],
        weight_gamma=config['train']['weight_gamma'],
        weight_index=config['train']['weight_index'],
        repeat_projector_num=config['train']['repeat_projector_num'],
        projector_dim=config['train']['projector_dim'],
    )
    val_criterion = SWDLoss(gamma=0.1)
    logger.info('create criterion successfully')
    generator = models.TransformerGenerator(
        latent_dim=config['data']['latent_dim'],
        quantities_num=config['data']['quantities_num'],
        embedding_num=config['model']['embedding_num'],
        nhead=config['model']['nhead'],
        dff=config['model']['dff'],
        norm=config['model']['norm'],
        activation=config['model']['activation'],
        num_layers=config['model']['num_layers'],
        cut_theta_index=config['model']['cut_index']).to(
            config['train']['device'])
    logger.info(
        summary(generator,
                input_data=torch.rand(1, config['data']['latent_dim']).to(
                    config['train']['device'])))
    logger.info('create model successfully')
    generator = torch.nn.parallel.DataParallel(generator)
    gen_optim, gen_schedule = make_optimizer(
        model=generator,
        name=config['optimizer']['name'],
        learning_rate=config['optimizer']['learning_rate'],
        warmup_steps=len(train_loader) *
        config['train']['epochs'] * config['optimizer']['warmup'],
        total_steps=len(train_loader) * config['train']['epochs'],
        decay_steps=config['optimizer']['decay_step'],
        val_loss_decay_time=config['optimizer']['val_loss_decay_time'],
        decay_type=config['optimizer']['decay_type'],
        num_cycles=config['optimizer']['num_cycles'],
        beta_1=config['optimizer']['beta_1'],
        beta_2=config['optimizer']['beta_2'],
        epsilon=config['optimizer']['epsilon'],
        weight_decay=config['optimizer']['weight_decay'])
    logger.info('create optimizer successfully')
    do_train = TrainModel(model=generator,
                          optimizer=gen_optim,
                          device=config['train']['device'],
                          expand_function=ExpandFreeQuantities(
                              decay_num=config['train']['decay_num'],
                              target_data=val_original_data,
                          ),
                          quantities_name=config['train']['quantities_name'],
                          output_dir=output_dir,
                          config=config,
                          latent_dim=config['data']['latent_dim'],
                          save_name=config['train']['save_name'],
                          loss=criterion,
                          val_loss=val_criterion)
    if config['model']['load_path'] is not None:
        do_train.load(
            os.path.join(config['output_dir'], config['model']['load_path']))
        logger.info('load model state dicts successfully')
    logger.info('begin to train')
    do_train.train(train_dataloader=train_loader,
                   train_logger=train_logger,
                   val_expand_data=val_expand_data,
                   val_free_data=val_free_data,
                   val_original_data=val_original_data,
                   schedule=gen_schedule,
                   val_logger=val_logger,
                   epochs=config['train']['epochs'],
                   use_tensorboard=config['train']['use_tensorboard'])
    logger.info('train successfully')
    torch.cuda.empty_cache()
    test_expand_data, test_free_data, test_original_data = get_validation_data(
        path=os.path.join(config['data']['datasets_path'],
                          config['data']['test_data_name']),
        decay_num=config['train']['decay_num'],
        dtype=config['data']['dtype'],
        distribution_transform=config['data']['distribution_transform'])
    test_logger.info('test data size is {}'.format(test_expand_data.shape[0]))
    do_train.test(test_logger=test_logger,
                  test_expand_data=test_expand_data,
                  test_free_data=test_free_data,
                  test_original_data=test_original_data)
    logger.info('test successfully')


class TrainModel:

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 expand_function,
                 quantities_name,
                 output_dir,
                 config,
                 save_name,
                 latent_dim,
                 val_loss,
                 device='cuda:0'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        self.quantities_name = quantities_name
        self.expand_function = expand_function
        self.output_dir = output_dir
        self.config = config
        self.random_choose_time = config['train']['random_choose_time']
        self.latent_dim = latent_dim
        self.save_name = save_name
        self.val_loss = val_loss

    def format_time(self, elapsed):
        elapsed_rounded = int(round(elapsed))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def train_step(self, inputs, free_targets, targets):
        self.model.train()
        loss = self.model(x=inputs,
                          expand_function=self.expand_function,
                          free_targets=free_targets,
                          targets=targets,
                          return_detail=True,
                          criterion=self.loss)
        total_loss = torch.mean(loss[0])
        wd_loss = torch.mean(loss[1])
        swd_loss = torch.mean(loss[2])
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), wd_loss.item(
        ), swd_loss.item()

    def test_step(self, inputs, random_projection=None):
        self.model.eval()
        inputs = inputs.to(self.device)
        expand_preds, free_preds, original_preds = self.model(
            x=inputs,
            expand_function=self.expand_function,
            return_all=True)
        if random_projection is not None:
            random_projection_expand_preds = torch.matmul(
                free_preds, random_projection)
            return expand_preds.detach().cpu(), free_preds.detach().cpu(
            ), original_preds.detach().cpu(
            ), random_projection_expand_preds.detach().cpu()
        else:
            return expand_preds.detach().cpu(), free_preds.detach().cpu(
            ), original_preds.detach().cpu(
            )

    def val_one_dimensional_plot(self, predicts, targets, permutation_list, epoch):
        epsilon = 1e-12
        predicts = predicts.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        ks_list = np.empty(len(self.quantities_name))
        for i in trange(len(self.quantities_name)):
            if i == 4:
                targets_hist, bins = np.histogram(
                    np.sqrt(
                        distribution_transform_fun(targets[:, i], 1.5, -0.6,
                                                   -312, 300)),
                    bins=60,
                    weights=np.ones_like(targets[:, i]) / targets[:, i].size)
                predicts_hist = np.histogram(
                    np.sqrt(
                        distribution_transform_fun(predicts[:, i], 1.5, -0.6,
                                                   -312, 300)),
                    bins=bins,
                    weights=np.ones_like(targets[:, i]) /
                    targets[:, i].size)[0]
            elif i == 5:
                targets_hist, bins = np.histogram(
                    np.sqrt(targets[:, i]),
                    bins=60,
                    weights=np.ones_like(targets[:, i]) / targets[:, i].size)
                predicts_hist = np.histogram(
                    np.sqrt(predicts[:, i]),
                    bins=bins,
                    weights=np.ones_like(targets[:, i]) /
                    targets[:, i].size)[0]
            else:
                targets_hist, bins = np.histogram(
                    targets[:, i],
                    bins=60,
                    weights=np.ones_like(targets[:, i]) / targets[:, i].size)
                predicts_hist = np.histogram(
                    predicts[:, i],
                    bins=bins,
                    weights=np.ones_like(targets[:, i]) /
                    targets[:, i].size)[0]
            predict_hists_error = np.sqrt(predicts_hist / predicts[:, i].size)
            targets_hists_error = np.sqrt(targets_hist / targets[:, i].size)
            hists_ratio_list = predicts_hist / (targets_hist + epsilon)
            hists_ratio_error_list = np.sqrt((predict_hists_error / (predicts_hist + epsilon)) ** 2 + (
                targets_hists_error / (targets_hist + epsilon)) ** 2) * hists_ratio_list
            ks_list[i] = kstest(predicts[:, i], targets[:, i])[1]
            ratio_plot(compare_hist=targets_hist,
                       hist_value=predicts_hist,
                       ratio_hist_value=hists_ratio_list,
                       bins=bins,
                       ks_test_value=ks_list[i],
                       xlabel=self.quantities_name[i],
                       title='Epoch:' + str(epoch),
                       ratio_hist_err=hists_ratio_error_list,
                       save_path=os.path.join(self.output_dir, 'val_fig',
                                              'quantities_' + str(i)),
                       save_name=str(epoch))
        return np.sum(ks_list > 0.05) / ks_list.size * 100

    def plot_val_loss_curve(self, values, ylabel, save_name):
        plot_line(values=values,
                  ylabel=ylabel,
                  save_path=os.path.join(
                      self.output_dir,
                      'curve_fig',
                  ),
                  save_name=save_name)

    def test_one_dimensional_plot(self, predicts, targets, permutation_list):
        epsilon = 1e-12
        predicts = predicts.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        ks_list = np.empty(len(self.quantities_name))
        wd_list = np.empty(len(self.quantities_name))
        for i in trange(len(self.quantities_name)):
            if i == 4:
                targets_hist, bins = np.histogram(
                    np.sqrt(
                        distribution_transform_fun(targets[:, i], 1.5, -0.6,
                                                   -312, 300)),
                    bins=60,
                    weights=np.ones_like(targets[:, i]) / targets[:, i].size)
                predicts_hist = np.histogram(
                    np.sqrt(
                        distribution_transform_fun(predicts[:, i], 1.5, -0.6,
                                                   -312, 300)),
                    bins=bins,
                    weights=np.ones_like(predicts[:, i]) /
                    predicts[:, i].size)[0]
            elif i == 5:
                targets_hist, bins = np.histogram(
                    np.sqrt(targets[:, i]),
                    bins=60,
                    weights=np.ones_like(targets[:, i]) / targets[:, i].size)
                predicts_hist = np.histogram(
                    np.sqrt(predicts[:, i]),
                    bins=bins,
                    weights=np.ones_like(predicts[:, i]) /
                    predicts[:, i].size)[0]
            else:
                targets_hist, bins = np.histogram(
                    targets[:, i],
                    bins=60,
                    weights=np.ones_like(targets[:, i]) / targets[:, i].size)
                predicts_hist = np.histogram(
                    predicts[:, i],
                    bins=bins,
                    weights=np.ones_like(predicts[:, i]) /
                    predicts[:, i].size)[0]
            predict_hists_error = np.sqrt(predicts_hist / predicts[:, i].size)
            targets_hists_error = np.sqrt(targets_hist / targets[:, i].size)
            hists_ratio_list = predicts_hist / (targets_hist + epsilon)
            hists_ratio_error_list = np.sqrt((predict_hists_error / (predicts_hist + epsilon)) ** 2 + (
                targets_hists_error / (targets_hist + epsilon)) ** 2) * hists_ratio_list
            ks_list[i] = kstest(predicts[:, i], targets[:, i])[1]
            wd_list[i] = self.val_loss.count_one_dimensional_p_value(
                predicts[:, i], targets[:, i], permutation_list=permutation_list)
            ratio_plot(compare_hist=targets_hist,
                       hist_value=predicts_hist,
                       ratio_hist_value=hists_ratio_list,
                       bins=bins,
                       ks_test_value=ks_list[i],
                       wd_test_value=wd_list[i],
                       xlabel=self.quantities_name[i],
                       ratio_hist_err=hists_ratio_error_list,
                       save_path=os.path.join(self.output_dir, 'test_fig'),
                       save_name=str(i))
        return np.sum(ks_list > 0.05) / ks_list.size * 100, np.sum(wd_list > 0.05) / wd_list.size * 100

    def test_one_dimensional_random_projection_plot(self, predicts, targets, permutation_list):
        predicts = predicts.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        ks_list = np.empty(predicts.shape[-1])
        wd_list = np.empty(predicts.shape[-1])
        for i in trange(predicts.shape[-1]):
            targets_hist, bins = np.histogram(
                targets[:, i],
                bins=60,
                weights=np.ones_like(targets[:, i]) / targets[:, i].size)
            predicts_hist = np.histogram(
                predicts[:, i],
                bins=bins,
                weights=np.ones_like(predicts[:, i]) /
                predicts[:, i].size)[0]
            predict_hists_error = np.sqrt(predicts_hist / predicts[:, i].size)
            targets_hists_error = np.sqrt(targets_hist / targets[:, i].size)
            hists_ratio_list = predicts_hist / (targets_hist + 1e-12)
            hists_ratio_error_list = np.sqrt((predict_hists_error / (predicts_hist + 1e-12)) ** 2 + (
                targets_hists_error / (targets_hist + 1e-12)) ** 2) * hists_ratio_list
            ks_list[i] = kstest(predicts[:, i], targets[:, i])[1]
            wd_list[i] = self.val_loss.count_one_dimensional_p_value(
                predicts[:, i], targets[:, i], permutation_list=permutation_list)
            ratio_plot(compare_hist=targets_hist,
                       hist_value=predicts_hist,
                       ratio_hist_value=hists_ratio_list,
                       bins=bins,
                       ks_test_value=ks_list[i],
                       wd_test_value=wd_list[i],
                       ratio_hist_err=hists_ratio_error_list,
                       xlabel='Random Projection' + str(i),
                       save_path=os.path.join(self.output_dir, 'test_fig'),
                       save_name='random_projection_' + str(i))
        return np.sum(ks_list > 0.05) / ks_list.size * 100, np.sum(wd_list > 0.05) / wd_list.size * 100

    def test_swd_test(self, expand_predicts, free_predicts, expand_targets,
                      free_targets, permutation_list):
        test_random_projection_generator = torch.Generator(device=self.device)
        test_random_projection_generator.manual_seed(1)
        test_random_projection = torch.randn(
            (free_targets.shape[1], 256),
            generator=test_random_projection_generator,
            device=self.device)
        test_random_projection /= torch.sqrt(
            torch.sum(test_random_projection ** 2, dim=0, keepdim=True))
        original_wd_loss, original_swd_loss, wd_p_value, swd_p_value = self.val_loss.count_p_value(
            free_predicts.to(self.device),
            expand_predicts.to(self.device),
            free_targets.to(self.device),
            expand_targets.to(self.device),
            permutation_list=permutation_list,
            random_projection=test_random_projection)
        return original_wd_loss, original_swd_loss, wd_p_value, swd_p_value

    def save_events(self, data):
        data = data.astype(np.float32)
        np.save(os.path.join(self.output_dir, 'Momentum_pipi.npy'), data)

    def do_one_epoch(self,
                     train_dataloader,
                     train_logger,
                     val_logger,
                     epoch, val_expand_data, val_free_data, val_original_data, permutation_list,
                     schedule=None):
        t0 = time.time()
        total_train_loss = 0
        total_wd_loss = 0
        total_swd_loss = 0
        for step, (inputs, free_targets,
                   targets) in enumerate(train_dataloader):
            step += 1
            loss, wd_loss, swd_loss = self.train_step(
                inputs, free_targets, targets)
            total_train_loss += loss
            total_wd_loss += wd_loss
            total_swd_loss += swd_loss
            if step % 100 == 0:
                elapsed = self.format_time(time.time() - t0)
                train_logger.info(
                    'Epoch:{}  Batch {:>5,}  of  {:>5,}. Elapsed: {:} Lr:{:.10f} Total Loss:{:.10f} WD Loss:{:.10f} SWD Loss:{:.10f}.'
                    .format(epoch, step, len(train_dataloader), elapsed,
                            self.optimizer.param_groups[0]['lr'],
                            total_train_loss / step,
                            total_wd_loss / step,
                            total_swd_loss / step))
            if schedule is not None:
                schedule.step()
        avg_train_loss = total_train_loss / step
        avg_wd_loss = total_wd_loss / step
        avg_swd_loss = total_swd_loss / step
        train_logger.info(
            'Epoch:{} Use Time:{:} Loss:{:.10f} WD Loss:{:.10f} SWD Loss:{:.10f}.'
            .format(epoch, self.format_time(time.time() - t0),
                    avg_train_loss, avg_wd_loss,
                    avg_swd_loss))
        with torch.no_grad():
            t0 = time.time()
            val_expand_predicts = []
            val_free_predicts = []
            val_original_predicts = []
            val_predicts_num = 0
            val_random_inputs_generator = torch.Generator()
            val_random_inputs_generator.manual_seed(2)
            while val_predicts_num <= val_expand_data.shape[0]:
                inputs = torch.rand(
                    (32 * torch.cuda.device_count(), self.latent_dim),
                    generator=val_random_inputs_generator) * 2 - 1
                expand_preds, free_preds, original_preds = self.test_step(
                    inputs)
                val_predicts_num += expand_preds.shape[0]
                val_expand_predicts.append(expand_preds)
                val_free_predicts.append(free_preds)
                val_original_predicts.append(original_preds)
            val_expand_predicts = torch.cat(val_expand_predicts,
                                            dim=0)[:val_expand_data.shape[0]]
            val_free_predicts = torch.cat(val_free_predicts,
                                          dim=0)[:val_expand_data.shape[0]]
            val_original_predicts = torch.cat(
                val_original_predicts, dim=0)[:val_expand_data.shape[0]]
            self.save_events(val_original_predicts.detach().cpu().numpy())
            val_one_dimensional_ks_test_ratio = self.val_one_dimensional_plot(
                val_expand_predicts, val_expand_data, permutation_list, epoch)
            val_wd_loss, val_swd_loss, val_wd_p_value, val_swd_p_value = self.test_swd_test(
                val_expand_predicts, val_free_predicts, val_expand_data, val_free_data, permutation_list=permutation_list)
            val_logger.info(
                'Epoch:{} Use Time:{:} Val One Dimensional ks test ratio:{:}%Val Wd Loss:{:.10f} Val Swd Loss:{:.10f} Val wd p value:{:.10f}% Val Swd p value:{:.10f}%'
                .format(epoch, self.format_time(time.time() - t0),
                        int(val_one_dimensional_ks_test_ratio), val_wd_loss, val_swd_loss, val_wd_p_value * 100, val_swd_p_value * 100))
        return avg_train_loss, val_one_dimensional_ks_test_ratio, val_wd_loss, val_swd_loss, val_wd_p_value, val_swd_p_value

    def train(
        self,
        train_dataloader,
        train_logger,
        val_logger,
        val_expand_data,
        val_free_data,
        val_original_data,
        schedule,
        epochs=100,
        use_tensorboard=False,
    ):
        if use_tensorboard:
            writer = SummaryWriter(self.output_dir)
        val_ks_test_ratio_list = []
        val_swd_loss_list = []
        val_wd_loss_list = []
        val_permutation_list_generator = torch.Generator(
            device='cpu')
        val_permutation_list_generator.manual_seed(2000)
        val_permutation_list = torch.stack([torch.randperm(
            val_expand_data.shape[0] * 2,
            generator=val_permutation_list_generator,
            device='cpu') for _ in range(1000)])
        for epoch in np.arange(epochs):
            avg_train_loss, val_one_dimensional_ks_test_ratio, val_wd_loss, val_swd_loss, val_wd_p_value, val_swd_p_value = self.do_one_epoch(
                train_dataloader, train_logger, val_logger, epoch, val_expand_data, val_free_data, val_original_data, val_permutation_list, schedule)
            val_swd_loss_list.append(val_swd_loss)
            val_wd_loss_list.append(val_wd_loss)
            val_ks_test_ratio_list.append(val_one_dimensional_ks_test_ratio)
            if epoch == 0:
                self.save()
                best_swd_loss = val_swd_loss
                best_swd_p_value = val_swd_p_value
                best_epoch = epoch
            else:
                if val_swd_loss < best_swd_loss:
                    self.save()
                    val_logger.info('Previous best loss is {:.10f}, Now best loss is {:.10f}'.format(
                        best_swd_loss, val_swd_loss))
                    best_swd_loss = val_swd_loss
                    best_swd_p_value = val_swd_p_value
                    best_epoch = epoch
                else:
                    val_logger.info('Now best swd loss is {:.10f}, best swd p value is {:.10f}, best epoch is {:}'.format(
                        best_swd_loss, best_swd_p_value, best_epoch))
            if use_tensorboard:
                writer.add_scalar('Loss/train', avg_train_loss, epoch)
                writer.add_scalar('WD_Loss/val',
                                  val_wd_loss, epoch)
                writer.add_scalar('SWD_Loss/val',
                                  val_swd_loss, epoch)
                writer.add_scalar('WD_P_value/val',
                                  val_wd_p_value, epoch)
                writer.add_scalar('SWD_P_value/val',
                                  val_swd_p_value, epoch)
        self.plot_val_loss_curve(
            val_swd_loss_list,
            ylabel='SWD Validation Loss',
            save_name='swd_validation_loss_curve')
        self.plot_val_loss_curve(
            val_wd_loss_list,
            ylabel='WD Validation Loss',
            save_name='wd_validation_loss_curve')

    def test(self, test_logger, test_expand_data,
             test_free_data,
             test_original_data,
             ):
        test_plot_random_projection_generator = torch.Generator(
            device=self.device)
        test_plot_random_projection_generator.manual_seed(1000)
        test_plot_random_projection = torch.randn(
            (test_free_data.shape[1],
             self.config['train']['plot_test_random_projection_num']),
            generator=test_plot_random_projection_generator,
            device=self.device)
        test_permutation_list_generator = torch.Generator(
            device='cpu')
        test_permutation_list_generator.manual_seed(3000)
        test_permutation_list = torch.stack([torch.randperm(
            test_expand_data.shape[0] * 2,
            generator=test_permutation_list_generator,
            device='cpu') for _ in range(1000)])
        test_random_projection_expand_data = torch.matmul(
            test_free_data,
            test_plot_random_projection.detach().cpu())
        self.load(os.path.join(self.output_dir, self.save_name))
        with torch.no_grad():
            t0 = time.time()
            test_expand_predicts = []
            test_free_predicts = []
            test_original_predicts = []
            test_random_projection_expand_predicts = []
            test_predicts_num = 0
            test_random_inputs_generator = torch.Generator()
            test_random_inputs_generator.manual_seed(3)
            while test_predicts_num <= test_expand_data.shape[
                    0]:
                inputs = torch.rand(
                    (32 * torch.cuda.device_count(), self.latent_dim),
                    generator=test_random_inputs_generator) * 2 - 1
                expand_preds, free_preds, original_preds, random_projection_expand_preds = self.test_step(
                    inputs, test_plot_random_projection)
                test_predicts_num += expand_preds.shape[0]
                test_expand_predicts.append(expand_preds)
                test_free_predicts.append(free_preds)
                test_original_predicts.append(original_preds)
                test_random_projection_expand_predicts.append(
                    random_projection_expand_preds)
            test_expand_predicts = torch.cat(
                test_expand_predicts, dim=0)[:test_expand_data.shape[0]]
            test_free_predicts = torch.cat(
                test_free_predicts, dim=0)[:test_expand_data.shape[0]]
            test_original_predicts = torch.cat(
                test_original_predicts, dim=0)[:test_expand_data.shape[0]]
            test_random_projection_expand_predicts = torch.cat(
                test_random_projection_expand_predicts,
                dim=0)[:test_expand_data.shape[0]]
            self.save_events(test_original_predicts.detach().cpu().numpy())
            one_dimensional_ks_test_ratio, one_dimensional_wd_test_ratio = self.test_one_dimensional_plot(
                test_expand_predicts, test_expand_data, test_permutation_list)
            random_projection_ks_test_ratio, random_projection_wd_test_ratio = self.test_one_dimensional_random_projection_plot(
                test_random_projection_expand_predicts,
                test_random_projection_expand_data, test_permutation_list)
            test_wd_loss, test_swd_loss, test_wd_p_value, test_swd_p_value = self.test_swd_test(
                test_expand_predicts, test_free_predicts, test_expand_data,
                test_free_data, test_permutation_list)
            total_test_save_dict = {
                'test_wd_loss': test_wd_loss,
                'test_swd_loss': test_swd_loss,
                'test_wd_p_value': test_wd_p_value,
                'test_swd_p_value': test_swd_p_value,
            }
            with open(os.path.join(self.output_dir, 'test_output.pickle'),
                      'wb') as f:
                pickle.dump(total_test_save_dict, f)
            test_logger.info(
                'Use Time:{:} One Dimensional Ks test Ratio:{:.10f}, One Dimensional Wd test Ratio:{:.10f}, Random Projection Ks test Ratio:{:.10f}, Random Projection Wd test Ratio:{:.10f}, Test Wd Loss:{:.10f} Test Swd Loss:{:.10f} Test wd p value:{:.10f}% Test Swd p value:{:.10f}%'
                .format(self.format_time(time.time() - t0),
                        one_dimensional_ks_test_ratio,
                        one_dimensional_wd_test_ratio,
                        random_projection_ks_test_ratio,
                        random_projection_wd_test_ratio,
                        test_wd_loss, test_swd_loss, test_wd_p_value * 100, test_swd_p_value * 100
                        ))

    def save(self):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output_dir, self.save_name))

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, help='choose config file')
    args = parser.parse_args()
    main(args.config_file)
