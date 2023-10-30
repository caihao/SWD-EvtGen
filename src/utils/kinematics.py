import torch
from torch.nn import Module
from torch import Tensor
import numpy as np

epsilon = 1e-6


class ExpandFreeQuantities(Module):

    def __init__(self, decay_num, target_data, distribution_transform=True):
        super().__init__()
        self.decay_num = decay_num
        self.expand_config = self.calculate_expand_config(target_data)
        self.f_invm_min = self.expand_config[6] + self.expand_config[7]
        self.distribution_transform = distribution_transform
        if distribution_transform:
            self.phi_invm_min = particle_att_utils.distribution_inv_transform(
                torch.tensor([1.006**2]), 1.5, -0.6, -312, 300)
            self.phi_invm_max = particle_att_utils.distribution_inv_transform(
                torch.tensor([1.032**2]), 1.5, -0.6, -312, 300)
        else:
            self.phi_invm_min = torch.tensor([1.006**2])
            self.phi_invm_max = torch.tensor([1.032**2])

    def calculate_expand_config(self, x: Tensor):
        psi_px = torch.mean(x[:, ::4].sum(dim=1))
        psi_py = torch.mean(x[:, 1::4].sum(dim=1))
        psi_pz = torch.mean(x[:, 2::4].sum(dim=1))
        psi_E = torch.mean(x[:, 3::4].sum(dim=1))
        metric = torch.tensor([-1, -1, -1, 1])
        kp1_invm = torch.mean(
            torch.sqrt((x[:, :4] * (x[:, :4] * metric)).sum(dim=1)))
        km1_invm = torch.mean(
            torch.sqrt((x[:, 4:8] * (x[:, 4:8] * metric)).sum(dim=1)))
        kp2_invm = torch.mean(
            torch.sqrt((x[:, 8:12] * (x[:, 8:12] * metric)).sum(dim=1)))
        km2_invm = torch.mean(
            torch.sqrt((x[:, 12:16] * (x[:, 12:16] * metric)).sum(dim=1)))
        config = torch.tensor([
            psi_px, psi_py, psi_pz, psi_E, kp1_invm, km1_invm, kp2_invm,
            km2_invm
        ],
            dtype=torch.float64)
        return config

    def expand_phi_f(self, x: Tensor, phi_invm_min: Tensor,
                     phi_invm_max: Tensor, f_invm_min: Tensor,
                     expand_config: Tensor):
        alpha = (x[:, 2] + 1) / 2
        phi_original_invm = phi_invm_min + alpha * (phi_invm_max -
                                                    phi_invm_min)
        if self.distribution_transform:
            phi_invm = particle_att_utils.distribution_transform(
                phi_original_invm, 1.5, -0.6, -312, 300)  # 实际invm的平方
        else:
            phi_invm = phi_original_invm
        f_invm_max = expand_config[3] - torch.sqrt(phi_invm)
        alpha_2 = (x[:, 3] + 1) / 2
        f_invm = f_invm_min + alpha_2 * (f_invm_max - f_invm_min)
        theta = torch.unsqueeze(x[:, 0], dim=1)  # 这里实际代表的是phi角
        Phi = torch.unsqueeze((x[:, 1] + 1) / 2, dim=1)  # 这里实际是theta角，即pz和p的夹角
        phi_p = torch.unsqueeze(particle_att_utils.pdk(
            expand_config[3], f_invm, torch.sqrt(phi_invm)),
            dim=1)
        phi_original_invm = torch.unsqueeze(phi_original_invm, dim=1)
        phi_invm = torch.unsqueeze(phi_invm, dim=1)
        f_invm = torch.unsqueeze(f_invm, dim=1)
        phi_pz = phi_p * torch.cos(torch.pi * Phi)
        phi_px = phi_p * torch.sin(torch.pi * Phi) * torch.cos(
            torch.pi * theta)
        phi_py = phi_p * torch.sin(torch.pi * Phi) * torch.sin(
            torch.pi * theta)
        phi_P = torch.cat([phi_px, phi_py, phi_pz], dim=1)
        phi_E = torch.sqrt(phi_invm +
                           torch.sum(phi_P**2, dim=1, keepdim=True) + epsilon)
        phi_momentum = torch.cat([phi_P, phi_E], dim=1)
        phi_pt = torch.sqrt(
            torch.sum(phi_momentum[:, :2]**2, dim=1, keepdim=True) + epsilon)
        phi_p = torch.sqrt(
            torch.sum(phi_momentum[:, :3]**2, dim=1, keepdim=True) + epsilon)
        phi_phi = torch.atan2(phi_momentum[:, 1], phi_momentum[:, 0])
        phi_theta = torch.acos(
            phi_momentum[:, 2] /
            torch.sqrt(phi_momentum[:, 0]**2 + phi_momentum[:, 1]**2 +
                       phi_momentum[:, 2]**2 + epsilon))
        angle_config = torch.stack([phi_theta, phi_phi], dim=1)
        return torch.cat([
            phi_momentum, phi_original_invm, f_invm**2, phi_pt, phi_p,
            angle_config
        ],
            dim=1)

    def expand_two_k(self, x, config, fmd):
        theta = torch.unsqueeze(x[:, 0], dim=1)
        # Phi = torch.unsqueeze((x[:, 1] + 1) / 2, dim=1)
        Phi = torch.unsqueeze(x[:, 1], dim=1)
        kp_P = torch.unsqueeze(config[:, 6], dim=1)
        kp_pz = kp_P * Phi
        kp_px = kp_P * torch.sqrt(1 - Phi**2 + epsilon) * torch.cos(
            torch.pi * theta)
        kp_py = kp_P * torch.sqrt(1 - Phi**2 + epsilon) * torch.sin(
            torch.pi * theta)
        kp_P = torch.cat([kp_px, kp_py, kp_pz], dim=1)
        km_P = -kp_P
        kp_momentum = torch.cat(
            [kp_P, torch.unsqueeze(config[:, 7], dim=1)], dim=1)
        km_momentum = torch.cat(
            [km_P, torch.unsqueeze(config[:, 7], dim=1)], dim=1)

        kp_rotation_momentum = particle_att_utils.Lorentz_inv_trans(
            fmd, kp_momentum)
        km_rotation_momentum = particle_att_utils.Lorentz_inv_trans(
            fmd, km_momentum)
        # kp_lorentz_transformation_momentum = particle_att_utils.lorentz_transformation(
        #     kp_momentum, config[:, 4])
        # km_lorentz_transformation_momentum = particle_att_utils.lorentz_transformation(
        #     km_momentum, config[:, 4])
        # kp_rotation_momentum = particle_att_utils.rotation(
        #     kp_lorentz_transformation_momentum, config[:, 0], config[:, 1],
        #     config[:, 2], config[:, 3])
        # km_rotation_momentum = particle_att_utils.rotation(
        #     km_lorentz_transformation_momentum, config[:, 0], config[:, 1],
        #     config[:, 2], config[:, 3])
        kp_pt = torch.sqrt(
            torch.sum(kp_rotation_momentum[:, :2]**2, dim=1, keepdim=True) +
            epsilon)
        km_pt = torch.sqrt(
            torch.sum(km_rotation_momentum[:, :2]**2, dim=1, keepdim=True) +
            epsilon)
        kp_phi = torch.atan2(kp_rotation_momentum[:, 1],
                             kp_rotation_momentum[:, 0])
        kp_theta = torch.acos(
            kp_rotation_momentum[:, 2] /
            torch.sqrt(kp_rotation_momentum[:, 0]**2 +
                       kp_rotation_momentum[:, 1]**2 +
                       kp_rotation_momentum[:, 2]**2 + epsilon))
        km_phi = torch.atan2(km_rotation_momentum[:, 1],
                             km_rotation_momentum[:, 0])
        km_theta = torch.acos(
            km_rotation_momentum[:, 2] /
            torch.sqrt(km_rotation_momentum[:, 0]**2 +
                       km_rotation_momentum[:, 1]**2 +
                       km_rotation_momentum[:, 2]**2 + epsilon))
        angle_config = torch.stack([kp_theta, kp_phi, km_theta, km_phi], dim=1)
        return torch.cat([
            kp_rotation_momentum, km_rotation_momentum, kp_pt, km_pt,
            angle_config
        ],
            dim=1)

    def calculate_mother_particle_config(self, mother_particle, expand_config, son_mass_1, son_mass_2):
        mother_invm = particle_att_utils.get_invm(mother_particle)
        beta = particle_att_utils.get_lorentz_transformation_static_to_moving_velocity(
            mother_particle)
        cz, sz, sy, cy = particle_att_utils.get_lorentz_static_to_moving_rotation_config(
            mother_particle)
        mother_config = torch.stack([cz, sz, sy, cy, beta], dim=1)
        son_static_P = particle_att_utils.pdk(
            torch.sqrt(mother_invm), son_mass_1,
            son_mass_2)
        son_static_E = particle_att_utils.get_static_E(
            torch.sqrt(mother_invm), son_mass_1, son_mass_2)
        mother_config = torch.cat(
            [mother_config, mother_invm, son_static_P, son_static_E], dim=1)
        return mother_config

    def expand_forward(self, x: Tensor):
        expand_config = self.expand_config.to(x.device)
        f_invm_min = self.f_invm_min.to(x.device)
        phi_invm_min = self.phi_invm_min.to(x.device)
        phi_invm_max = self.phi_invm_max.to(x.device)
        if self.decay_num == 0:
            return x
        phi_f_config = self.expand_phi_f(x[:, :4], phi_invm_min, phi_invm_max,
                                         f_invm_min, expand_config)
        if self.decay_num == 1:
            return phi_f_config
        # phi_f_config :[phi_momentum(px,py,pz,E), phi_original_invm, f_invm ** 2, phi_pt, phi_p, angle_config]
        # expand_config :psi_px, psi_py, psi_pz, psi_E, kp1_invm, km1_invm, kp2_invm, km2_invm
        f = torch.cat([
            -phi_f_config[:, :3],
            (self.expand_config[3] - phi_f_config[:, 3]).unsqueeze(dim=1)
        ],
            dim=1)
        f_config = self.calculate_mother_particle_config(
            f, expand_config, expand_config[6], expand_config[7])
        kp2_km2_config = self.expand_two_k(x[:, 4:6], f_config, f)
        if self.decay_num == 2:
            return torch.cat([phi_f_config, kp2_km2_config], dim=1)
        phi_config = self.calculate_mother_particle_config(
            phi_f_config[:, :4], expand_config, expand_config[4], expand_config[5])
        kp1_km1_config = self.expand_two_k(x[:, 6:8], phi_config,
                                           phi_f_config[:, :4])
        if self.decay_num == 3:
            return torch.cat([phi_f_config, kp2_km2_config, kp1_km1_config],
                             dim=1)

    def forward(self, x, return_four_momentum=False):
        x = self.expand_forward(x)
        if self.decay_num == 3 and return_four_momentum:
            return torch.cat([x[:, 24:32], x[:, 10:18]], dim=1)
        else:
            return x


class ExpandFourMomentumQuantities(Module):

    def __init__(self, decay_num, distribution_transform):
        super().__init__()
        self.decay_num = decay_num
        self.distribution_transform = distribution_transform

    def calculate_x_kp1_km1(self, x):
        kp1_rotation_momentum = x[:, :4]
        km1_rotation_momentum = x[:, 4:8]
        kp1_pt = np.sqrt(
            np.sum(kp1_rotation_momentum[:, :2]**2, axis=1, keepdims=True))
        km1_pt = np.sqrt(
            np.sum(km1_rotation_momentum[:, :2]**2, axis=1, keepdims=True))
        kp1_phi = np.arctan2(kp1_rotation_momentum[:, 1],
                             kp1_rotation_momentum[:, 0])
        kp1_theta = np.arccos(kp1_rotation_momentum[:, 2] /
                              np.sqrt(kp1_rotation_momentum[:, 0]**2 +
                                      kp1_rotation_momentum[:, 1]**2 +
                                      kp1_rotation_momentum[:, 2]**2))
        km1_phi = np.arctan2(km1_rotation_momentum[:, 1],
                             km1_rotation_momentum[:, 0])
        km1_theta = np.arccos(km1_rotation_momentum[:, 2] /
                              np.sqrt(km1_rotation_momentum[:, 0]**2 +
                                      km1_rotation_momentum[:, 1]**2 +
                                      km1_rotation_momentum[:, 2]**2))
        angle_config = np.stack([kp1_theta, kp1_phi, km1_theta, km1_phi],
                                axis=1)
        return np.concatenate([
            kp1_rotation_momentum, km1_rotation_momentum, kp1_pt, km1_pt,
            angle_config
        ],
            axis=1)

    def calculate_target_data(self, x):
        phi = x[:, :4] + x[:, 4:8]
        phi_invm = calculate_invm_2d(phi)
        if self.distribution_transform:
            phi_invm = distribution_inv_transform_fun(phi_invm, 1.5, -0.6,
                                                      -312, 300)
        f = x[:, 8:12] + x[:, 12:]
        f_invm = calculate_invm_2d(f)
        phi_momentum = x[:, :4] + x[:, 4:8]
        kp2_momentum = x[:, 8:12]
        km2_momentum = x[:, 12:16]
        phi_pt = np.sqrt(np.sum(phi_momentum[:, :2]**2, axis=1, keepdims=True))
        kp2_pt = np.sqrt(np.sum(kp2_momentum[:, :2]**2, axis=1, keepdims=True))
        km2_pt = np.sqrt(np.sum(km2_momentum[:, :2]**2, axis=1, keepdims=True))
        phi_p = np.sqrt(np.sum(phi_momentum[:, :3]**2, axis=1, keepdims=True))
        phi_phi = np.arctan2(phi_momentum[:, 1], phi_momentum[:, 0])
        phi_theta = np.arccos(
            phi_momentum[:, 2] /
            np.sqrt(phi_momentum[:, 0]**2 + phi_momentum[:, 1]**2 +
                    phi_momentum[:, 2]**2))
        kp2_phi = np.arctan2(kp2_momentum[:, 1], kp2_momentum[:, 0])
        kp2_theta = np.arccos(
            kp2_momentum[:, 2] /
            np.sqrt(kp2_momentum[:, 0]**2 + kp2_momentum[:, 1]**2 +
                    kp2_momentum[:, 2]**2))
        km2_phi = np.arctan2(km2_momentum[:, 1], km2_momentum[:, 0])
        km2_theta = np.arccos(
            km2_momentum[:, 2] /
            np.sqrt(km2_momentum[:, 0]**2 + km2_momentum[:, 1]**2 +
                    km2_momentum[:, 2]**2))
        angle_config = np.stack(
            [phi_theta, phi_phi, kp2_theta, kp2_phi, km2_theta, km2_phi],
            axis=1)
        if self.decay_num == 1:
            return np.concatenate([
                phi_momentum, phi_invm, f_invm, phi_pt, phi_p,
                angle_config[:, :2]
            ],
                axis=1)
        if self.decay_num >= 2:
            return np.concatenate([
                phi_momentum, phi_invm, f_invm, phi_pt, phi_p,
                angle_config[:, :2], kp2_momentum, km2_momentum, kp2_pt,
                km2_pt, angle_config[:, 2:]
            ],
                axis=1)

    def forward(self, x):
        if self.decay_num == 0:
            free_quantities_obj = GetFreeQuantities(x)
            return free_quantities_obj()
        if self.decay_num == 1 or self.decay_num == 2:
            return self.calculate_target_data(x)
        if self.decay_num == 3:
            return np.concatenate(
                [self.calculate_target_data(x),
                 self.calculate_x_kp1_km1(x)],
                axis=1)


class GetFreeQuantities(Module):

    def __init__(self, fourMomentumData,
                 distribution_transform):  # data->data of four momentum
        super().__init__()
        self.fourMomentumData = torch.from_numpy(fourMomentumData)
        self.distribution_transform = distribution_transform

    def getFreeAttr(
            self):  # get some free quantities from four momentum data first
        # fmd means four momentum data
        # four momentum data of the Kplus and Kminus from Phi particle
        x = self.fourMomentumData[:, :8]
        y = self.fourMomentumData[:, 8:16]
        # four momentum data (fmd) of the Phi and F particle
        self.Phi_Kp_fmd = x[:, :4]
        self.F_Kp_fmd = y[:, :4]
        self.Phi_fmd = torch.stack([
            x[:, ::4].sum(dim=1), x[:, 1::4].sum(dim=1), x[:, 2::4].sum(dim=1),
            x[:, 3::4].sum(dim=1)
        ],
            dim=1)
        self.F_fmd = torch.stack([
            y[:, ::4].sum(dim=1), y[:, 1::4].sum(dim=1), y[:, 2::4].sum(dim=1),
            y[:, 3::4].sum(dim=1)
        ],
            dim=1)

        self.Phi_theta = particle_att_utils.get_theta(
            self.Phi_fmd)  # 因为对接问题，这个theta角代表Pz和P的夹角，将被储存在第二列
        self.Phi_phi = particle_att_utils.get_phi(self.Phi_fmd)
        self.Phi_invm = torch.sqrt(particle_att_utils.get_invm(self.Phi_fmd))

        kp2_invm = torch.mean(torch.sqrt(particle_att_utils.get_invm(
            y[:, :4])))
        km2_invm = torch.mean(
            torch.sqrt(particle_att_utils.get_invm(y[:, 4:8])))
        self.F_invm_min = kp2_invm + km2_invm
        Psi_invm = torch.sum(
            self.fourMomentumData[:, 3::4], dim=1, keepdim=True)
        self.F_invm_max = Psi_invm - self.Phi_invm

        self.F_invm = torch.sqrt(particle_att_utils.get_invm(self.F_fmd))

    def conver(self):  # transform the free quantities to (-1,1)

        self.Phi_theta_tran = 2 * (
            self.Phi_theta / torch.pi) - 1  # (0,pi) -> (0,1) -> (0,2) -> (-1,1)
        self.Phi_phi_tran = self.Phi_phi / torch.pi  # (-pi,pi) -> (-1,1)

        if self.distribution_transform:
            Phi_invm_DisTran = particle_att_utils.distribution_inv_transform(
                self.Phi_invm**2, 1.5, -0.6, -312, 300)
        else:
            Phi_invm_DisTran = self.Phi_invm**2
        # get the maximun and minimun of Phi_invm which has been distribution transformed
        if self.distribution_transform:
            Phi_invm_max = particle_att_utils.distribution_inv_transform(
                torch.tensor(1.032**2), 1.5, -0.6, -312, 300)
            Phi_invm_min = particle_att_utils.distribution_inv_transform(
                torch.tensor(1.006**2), 1.5, -0.6, -312, 300)
        else:
            Phi_invm_max = torch.tensor(1.032**2)
            Phi_invm_min = torch.tensor(1.006**2)
        self.Phi_invm_tran = ((Phi_invm_DisTran - Phi_invm_min) /
                              (Phi_invm_max - Phi_invm_min)) * 2 - 1

        self.F_invm_tran = ((self.F_invm - self.F_invm_min) /
                            (self.F_invm_max - self.F_invm_min)) * 2 - 1

        Kp_at_F_fmd = particle_att_utils.Lorentz_trans(
            self.F_fmd, self.F_Kp_fmd)  # four monmentum data for Kp at f Frame
        Kp_at_Phi_fmd = particle_att_utils.Lorentz_trans(
            self.Phi_fmd, self.Phi_Kp_fmd)

        # Kp_theta_at_F = particle_att_utils.get_theta(Kp_at_F_fmd)
        Kp_phi_at_F = particle_att_utils.get_phi(Kp_at_F_fmd)
        Kp_F_pz = torch.unsqueeze(Kp_at_F_fmd[:, 2], dim=1)
        self.Kp_theta_at_F_trans = Kp_F_pz / particle_att_utils.get_p(
            Kp_at_F_fmd)
        self.Kp_phi_at_F_trans = Kp_phi_at_F / torch.pi

        # Kp_theta_at_Phi = particle_att_utils.get_theta(Kp_at_Phi_fmd)
        Kp_phi_at_Phi = particle_att_utils.get_phi(Kp_at_Phi_fmd)
        Kp_Phi_pz = torch.unsqueeze(Kp_at_Phi_fmd[:, 2], dim=1)
        self.Kp_theta_at_Phi_trans = Kp_Phi_pz / particle_att_utils.get_p(
            Kp_at_Phi_fmd)
        self.Kp_phi_at_Phi_trans = Kp_phi_at_Phi / torch.pi

    def forward(self):
        self.getFreeAttr()
        self.conver()
        feature = torch.stack([
            self.Phi_phi_tran, self.Phi_theta_tran, self.Phi_invm_tran,
            self.F_invm_tran, self.Kp_phi_at_F_trans, self.Kp_theta_at_F_trans,
            self.Kp_phi_at_Phi_trans, self.Kp_theta_at_Phi_trans
        ],
            dim=1)
        return feature.squeeze(dim=-1).detach().cpu().numpy()


def calculate_invm_2d(momentum):
    return (momentum[:, 3]**2 - np.sum(momentum[:, :3]**2, axis=1))[:, None]


def distribution_transform_fun(x, d, e, a, b): return (
    (np.sinh(d * np.arcsinh(x) - e)) - a) / b


def distribution_inv_transform_fun(x, d, e, a, b): return np.sinh(
    (e + np.arcsinh(b * x + a)) / d)


class particle_att_utils:

    @staticmethod
    def get_theta(momentum):
        return torch.arccos(momentum[:, 2][:, None] /
                            particle_att_utils.get_p(momentum))

    @staticmethod
    def get_phi(momentum):
        return torch.atan2(momentum[:, 1], momentum[:, 0])[:, None]

    @staticmethod
    def get_p(momentum):
        return torch.sqrt(
            torch.sum(momentum[:, :3]**2, dim=1, keepdim=True) + epsilon)

    @staticmethod
    def get_pt(momentum):
        return torch.sqrt(
            torch.sum(momentum[:, :2]**2, dim=1, keepdim=True) + epsilon)

    @staticmethod
    def get_P(momentum):
        return momentum[:, :3]

    @staticmethod
    def get_E(momentum):
        return momentum[:, -1][:, None]

    @staticmethod
    def get_invm(momentum):
        return momentum[:, 3][:, None]**2 - torch.sum(
            momentum[:, :3]**2, dim=1, keepdim=True)

    @staticmethod
    def get_all_att(momentum):
        return particle_att_utils.get_theta(
            momentum), particle_att_utils.get_phi(
                momentum), particle_att_utils.get_p(
                    momentum), particle_att_utils.get_pt(
                        momentum), particle_att_utils.get_P(
                            momentum), particle_att_utils.get_E(momentum)

    @staticmethod
    def get_all_att_include_invm(momentum):
        return particle_att_utils.get_theta(
            momentum
        ), particle_att_utils.get_phi(momentum), particle_att_utils.get_p(
            momentum), particle_att_utils.get_pt(
                momentum), particle_att_utils.get_P(
                    momentum), particle_att_utils.get_E(
                        momentum), particle_att_utils.get_invm(momentum)

    @staticmethod
    def get_lorentz_transformation_static_to_moving_velocity(
            momentum):  # 返回值是负数
        return -(particle_att_utils.get_p(momentum) /
                 (particle_att_utils.get_E(momentum) + epsilon))[:, 0]

    @staticmethod
    def get_lorentz_static_to_moving_rotation_config(momentum):
        """
        p = particle_att_utils.get_p(momentum)
        cz = momentum[:, 1][:, None] / p
        sz = torch.sqrt(1 - cz ** 2)
        sy = - momentum[:, 2][:, None] / (sz * p)
        cy = - momentum[:, 0][:, None] / (sz * p)
        """
        p = particle_att_utils.get_p(momentum)[:, 0]
        cz = momentum[:, 1] / (p + epsilon)
        sz = torch.sqrt(1 - cz**2 + epsilon)
        sy = -momentum[:, 2] / (sz * p + epsilon)
        cy = -momentum[:, 0] / (sz * p + epsilon)
        return cz, sz, sy, cy

    @staticmethod
    def lorentz_transformation(momentum, v):
        beta = v
        gamma = 1 / torch.sqrt(1 - beta**2 + epsilon)
        E = gamma * (momentum[:, 3] - beta * momentum[:, 1])
        Py = gamma * (momentum[:, 1] - beta * momentum[:, 3])
        new_momentum = torch.stack([momentum[:, 0], Py, momentum[:, 2], E],
                                   dim=1)
        return new_momentum

    @staticmethod
    def rotation(momentum, cz, sz, sy, cy):
        px = cz * momentum[:, 0] - sz * momentum[:, 1]
        py = sz * momentum[:, 0] + cz * momentum[:, 1]
        pz = sy * px + cy * momentum[:, 2]
        px = cy * px - sy * momentum[:, 2]
        new_momentum = torch.stack([px, py, pz, momentum[:, 3]], dim=1)
        return new_momentum

    @staticmethod
    def pdk(a, b, c):
        lam = (a + b + c) * (a - b + c) * (a + b - c) * (a - b - c)
        return torch.sqrt(lam + epsilon) / (2 * a + epsilon)

    @staticmethod
    def get_static_E(a, b, c):
        return (a**2 + b**2 - c**2) / (2 * a + epsilon)

    @staticmethod
    def distribution_transform(x, d, e, a, b):
        return ((torch.sinh(d * torch.arcsinh(x) - e)) - a) / (b + epsilon)

    @staticmethod
    def distribution_inv_transform(x, d, e, a, b):
        return torch.sinh((e + torch.arcsinh(b * x + a)) / (d + epsilon))

    @staticmethod
    def Lorentz_trans(f_phi_fmd, Kp_fmd):
        # f_phi_fmd is the four_monmentum_data of our purpose center-of-mass frame
        E = particle_att_utils.get_E(f_phi_fmd)[:, 0]
        vx = f_phi_fmd[:, 0] / (E + epsilon)
        vy = f_phi_fmd[:, 1] / (E + epsilon)
        vz = f_phi_fmd[:, 2] / (E + epsilon)
        px0 = Kp_fmd[:, 0]
        py0 = Kp_fmd[:, 1]
        pz0 = Kp_fmd[:, 2]
        E0 = Kp_fmd[:, 3]

        v = -particle_att_utils.get_lorentz_transformation_static_to_moving_velocity(
            f_phi_fmd)
        # beta = v
        gamma = 1 / torch.sqrt(1 - v**2 + epsilon)

        px = (1 + (gamma - 1) * (vx**2 / (v**2 + epsilon))) * px0 + (
            gamma - 1) * (vx * vy / (v**2 + epsilon)) * py0 + (gamma - 1) * (
                vx * vz / (v**2 + epsilon)) * pz0 - gamma * vx * E0
        py = (1 + (gamma - 1) * (vy**2 / (v**2 + epsilon))) * py0 + (
            gamma - 1) * (vy * vz / (v**2 + epsilon)) * pz0 + (gamma - 1) * (
                vy * vx / (v**2 + epsilon)) * px0 - gamma * vy * E0
        pz = (1 + (gamma - 1) * (vz**2 / (v**2 + epsilon))) * pz0 + (
            gamma - 1) * (vz * vx / (v**2 + epsilon)) * px0 + (gamma - 1) * (
                vz * vy / (v**2 + epsilon)) * py0 - gamma * vz * E0
        E = -gamma * (vx * px0 + vy * py0 + vz * pz0 - E0)
        return torch.stack([px, py, pz, E], dim=1)

    def Lorentz_inv_trans(f_phi_fmd, Kp_fmd):
        # f_phi_fmd is the four_monmentum_data of our purpose center-of-mass frame
        E = particle_att_utils.get_E(f_phi_fmd)[:, 0]
        vx = f_phi_fmd[:, 0] / (E + epsilon)
        vy = f_phi_fmd[:, 1] / (E + epsilon)
        vz = f_phi_fmd[:, 2] / (E + epsilon)
        px0 = Kp_fmd[:, 0]
        py0 = Kp_fmd[:, 1]
        pz0 = Kp_fmd[:, 2]
        E0 = Kp_fmd[:, 3]

        v = -particle_att_utils.get_lorentz_transformation_static_to_moving_velocity(
            f_phi_fmd)
        # beta = v
        gamma = 1 / torch.sqrt(1 - v**2 + epsilon)

        px = (1 + (gamma - 1) *
              (vx**2 / (v**2 + epsilon))) * px0 + (gamma - 1) * (
                  (vx * vy) / (v**2 + epsilon)) * py0 + (gamma - 1) * (
                      (vx * vz) / (v**2 + epsilon)) * pz0 + gamma * vx * E0
        py = (1 + (gamma - 1) *
              (vy**2 / (v**2 + epsilon))) * py0 + (gamma - 1) * (
                  (vy * vz) / (v**2 + epsilon)) * pz0 + (gamma - 1) * (
                      (vy * vx) / (v**2 + epsilon)) * px0 + gamma * vy * E0
        pz = (1 + (gamma - 1) *
              (vz**2 / (v**2 + epsilon))) * pz0 + (gamma - 1) * (
                  (vz * vx) / (v**2 + epsilon)) * px0 + (gamma - 1) * (
                      (vz * vy) / (v**2 + epsilon)) * py0 + gamma * vz * E0
        E = gamma * (vx * px0 + vy * py0 + vz * pz0 + E0)
        return torch.stack([px, py, pz, E], dim=1)


if __name__ == '__main__':
    import numpy as np

    device = 'cuda:1'
    torch.set_default_dtype(torch.float64)
    target_data = np.load('../../datasets/Momentum_kk.npy').astype(np.float64)

    getFreeQuan = GetFreeQuantities(target_data)
    freequan = np.squeeze(getFreeQuan())
    np.save('../../datasets/Momentum_kk_free.npy', freequan)
    """
    x = torch.tanh(torch.randn(4, 8, 1024, dtype=torch.float64))
    expand_obj = ExpandFreeQuantities(3, target_data, device='cpu')
    result = expand_obj(x, return_four_momentum=True)
    np.save('result.npy', result.detach().cpu().numpy())

    free_result = expand_obj(freequan, return_four_momentum=True)
    np.save('free_result.npy', free_result.detach().cpu().numpy())
    """

    # re_fmd = expand_obj(freequan.reshape(500000,8), return_four_momentum=True)
    # np.save('tmp.npy', expand_obj(freequan, return_four_momentum=True))
    # print(re_fmd.shape)
    # print(target_data.shape)
    # print(freequan.shape)
    # np.savetxt('3.txt', np.array(re_fmd.reshape(500000,16)-target_data))
    # np.savetxt('target_data.txt', np.array(target_data))
    # np.savetxt('re_fdm.txt', np.array(re_fmd.reshape(500000,16)))
    # np.savetxt('frequan.txt', np.array(freequan.reshape(500000,8)))
