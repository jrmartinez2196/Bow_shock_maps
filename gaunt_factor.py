import numpy as np
from pathlib import Path
from constants import Ry, kB, h

class GauntFactor:
    def __init__(self, data_file='gauntff.dat'):
        self.nu_mu = 146
        self.nu_mga = 81
        self.log_gamma2_start = -6.0
        self.log_u_start = -16.0
        self.step = 0.2
        self.gff = self._load_table(data_file)

    def _load_table(self, data_file):
        file_path = Path(data_file)
        if not file_path.exists():
            file_path = Path(__file__).parent / data_file

        if not file_path.exists():
            raise FileNotFoundError(f"Gaunt factor data file not found: {data_file}")

        with open(file_path, 'r') as f:
            lines = f.readlines()

        data_lines = lines[16:]

        gff = np.zeros((self.nu_mu, self.nu_mga))
        for i in range(self.nu_mu):
            gff[i, :] = np.fromstring(data_lines[i], sep=' ')[:self.nu_mga]

        return gff

    def gaunt_ff_calc(self, nu_ff, T, Z):
        nu_ff = np.asarray(nu_ff, dtype=float)
        T = np.asarray(T, dtype=float)

        # Broadcasting
        nu_ff, T = np.broadcast_arrays(nu_ff, T)

        # Avoid numerical errors
        valid = (T > 0) & np.isfinite(T)

        result = np.ones_like(T)

        if not np.any(valid):
            return result

        T_valid = T[valid]
        nu_valid = nu_ff[valid]

        gamma2 = Z**2 * Ry / (kB * T_valid)
        log_gamma2 = np.log10(gamma2)

        mu = h * nu_valid / (kB * T_valid)
        log_mu = np.log10(mu)

        # Table indexes
        jg = ((log_gamma2 + 6.0) / self.step).astype(int)
        jm = ((log_mu + 16.0) / self.step).astype(int)

        # Clipping within a valid range
        jg = np.clip(jg, 0, self.nu_mga - 2)
        jm = np.clip(jm, 0, self.nu_mu - 2)

        dg = (log_gamma2 + 6.0) / self.step - jg
        dm = (log_mu + 16.0) / self.step - jm

        # Interpolate
        f00 = self.gff[jm, jg]
        f10 = self.gff[jm, jg + 1]
        f01 = self.gff[jm + 1, jg]

        fx = (f10 - f00)
        fy = (f01 - f00)

        g_interp = f00 + fx * dg + fy * dm

        result[valid] = g_interp

        return result


# Global instance
_gaunt_instance = None

def gaunt_ff_calc(nu_ff, T_l, Zq_l, data_file='gauntff.dat'):
    global _gaunt_instance
    if _gaunt_instance is None:
        _gaunt_instance = GauntFactor(data_file)
    return _gaunt_instance.gaunt_ff_calc(nu_ff, T_l, Zq_l)