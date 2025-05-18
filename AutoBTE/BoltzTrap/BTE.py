import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from BoltzTraP2 import bandlib as BL
from BoltzTraP2 import dft as BTP
from BoltzTraP2 import fite, serialization, sphere, units
from ase.io import read
from ase.dft.kpoints import bandpath, get_special_points
from datetime import datetime
import threading
import time

class btp2:
    def __init__(self, data_path: str):
        '''
        Initialize the btp2 class.

        Args:
            data_path (str): The path to the directory containing the VASP output files.
        '''
        self.current_path = os.getcwd()
        self.data_path    = os.path.join(self.current_path, data_path)
        self.check_outcar_file()
        self.bt2file = os.path.join(self.data_path, 'interpolation.bt2')

    def check_outcar_file(self):
        '''
        Check if 'OUTCAR' file exists at the data_path. If not, raise an error.
        '''
        outcar_path = os.path.join(self.data_path, 'OUTCAR')
        if not os.path.isfile(outcar_path):
            raise FileNotFoundError(f"The file 'OUTCAR' does not exist at the specified path: {self.data_path}")
        
    def get_bt2file(self):
        '''
        Get the path of interpolation file(.bt2).
        '''
        return self.bt2file
    
    def set_bt2file(self, bt2file_path: str):
        '''
        Set the path of interpolation file manually.
        '''
        self.bt2file = bt2file_path

    @staticmethod
    def check_numpy_version():
        '''
        Check the version of numpy currently installed.
        '''
        version = np.__version__
        print(f'Numpy version : {version}')
        return version

    def interpolate(self, n=5, **kwargs):
        '''
        Interpolate between calculated k points and save the interpolated result at the given path.
        This method generates an interpolation of the band structure based on the provided k-point data. If an interpolation file 
        already exists at the specified path, the method will load the existing interpolation results unless forced to re-interpolate.

        Args:
            n (int): Sets up a k-point grid with roughly `n` times the density of the input k-points.
                     Default value is 5.
            eq (int): Number of equivalent k points to consider.
            emin (float): Minimum energy (in eV) to consider for interpolation.
            emax (float): Maximum energy (in eV) to consider for interpolation.
            rmin (float): Relative minimum energy (in eV) relative to the Fermi level. 
                          If specified, `emin` should not be provided.
            rmax (float): Relative maximum energy (in eV) relative to the Fermi level. 
                          If specified, `emax` should not be provided.
            restart (bool): If `True`, forces re-interpolation even if an existing interpolation file is found.
                            Default is `False`.
        '''
        if 'rmin' in kwargs and 'emin' in kwargs:
            raise ValueError("Both 'rmin' and 'emin' cannot be provided at the same time.")
        if 'rmax' in kwargs and 'emax' in kwargs:
            raise ValueError("Both 'rmax' and 'emax' cannot be provided at the same time.")
        
        if 'eq' in kwargs:
            eq = kwargs['eq']
            # build a new kwargs that no longer contains 'eq'
            reduced_kwargs = {k: v for k, v in kwargs.items() if k != 'eq'}

            # first recursion with fixed n=10
            self.interpolate(n=10, **reduced_kwargs)

            # compute a second n from eq and equivalences
            newn = int(eq / len(self.equivalences) * 10)
            self.interpolate(n=newn, **reduced_kwargs)

            return
        restart = kwargs.get('restart', False)
        interpolate_needed = False

        if not os.path.exists(self.bt2file) or restart:
            print('No existing interpolation results or restart requested.')
            interpolate_needed = True
        else:
            print('Existing interpolation results found.')
            data, existing_equivalences, _, _ = serialization.load_calculation(self.bt2file)
            DFTdata = BTP.DFTData(self.data_path)
            new_equivalences = sphere.get_equivalences(DFTdata.atoms, DFTdata.magmom, len(DFTdata.kpoints) * n)

            if len(existing_equivalences) != len(new_equivalences):
                print('Existing equivalences length mismatch. Re-interpolating.')
                interpolate_needed = True

        if interpolate_needed:
            DFTdata = BTP.DFTData(self.data_path)
            emin = kwargs.get('emin', -float('inf'))
            emax = kwargs.get('emax', float('inf'))

            if 'rmin' in kwargs:
                emin = DFTdata.fermi - (kwargs.get('rmin') * units.eV)
            if 'rmax' in kwargs:
                emax = DFTdata.fermi + (kwargs.get('rmax') * units.eV)
            if 'emin' in kwargs:
                emin *= units.eV
            if 'emax' in kwargs:
                emax *= units.eV

            DFTdata.bandana(emin=emin, emax=emax)
            equivalences = sphere.get_equivalences(DFTdata.atoms, DFTdata.magmom, len(DFTdata.kpoints) * n)
            coeffs = fite.fitde3D(DFTdata, equivalences)

            serialization.save_calculation(
                self.bt2file,
                DFTdata,
                equivalences,
                coeffs,
                serialization.gen_bt2_metadata(DFTdata, DFTdata.mommat is not None),
            )

            print(f'Interpolation results saved at {self.bt2file}')

        else:
            print('Using existing interpolation results.')
    
        data, equivalences, coeffs, metadata = serialization.load_calculation(self.bt2file)
        self.data = data
        self.equivalences = equivalences
        self.coeffs = coeffs
        self.metadata = metadata
        self.fermi = data.fermi / units.eV  # [eV]
        self.lattvec = data.get_lattvec()
        self.DOS = self._DOS(self)
        self.BandStructure = self._BandStructure(self)
        self.FermiIntegrals = self._FermiIntegrals(self)

    def _check_interpolation_status(self):
        """
        Check if the interpolation has been done either in memory or by checking for the interpolated file.
        """
        if hasattr(self, 'coeffs') and self.coeffs is not None:
            return True  

        if os.path.exists(self.get_bt2file()):
            try:
                data, equivalences, coeffs, metadata = serialization.load_calculation(self.get_bt2file())
                self.data = data
                self.equivalences = equivalences
                self.coeffs = coeffs
                self.metadata = metadata
                self.fermi = data.fermi / units.eV  # [eV]
                self.lattvec = data.get_lattvec()
                return True
            except Exception as e:
                print(f"Failed to load interpolation data from file: {e}")
        
        return False

    def _ensure_interpolation(self):
        if not self._check_interpolation_status():
            print("Interpolation not found. Performing interpolation in default settings...")
            self.interpolate()
        else:
            pass
    
    def set_fermi_energy(self, energy):
        '''
        Set the fermi energy.

        Args:
            energy (float): Fermi energy (eV) to set.
        '''
        self.fermi = energy

    def get_fermi_energy(self, method='vasp', update=False, **kwargs):
        '''
        Get the Fermi energy using the specified method.

        Args:
            method (str): Method to use for calculating Fermi energy. 
                        Options are:
                        - 'vasp': Use the Fermi energy directly from VASP output (default).
                        - 'refined': Calculate Fermi energy by refining the chemical potential.
                                     First, a value of mu is obtained that yields a number of electrons as close as possible to N0. 
                                     If mu falls in a wide gap (relative to kB * T), then mu0 is moved to the center of the gap.
                        - 'egrid': Calculate Fermi energy based on the energy grid.
            update (bool): If True, update `self.fermi` with the calculated value. Default is False.
            
        Kwargs:
        (1) 'refined'
            T (float): Temperature in Kelvin. Default is 300 K.
            refine (bool): If True, refine the Fermi energy. Default is False.
            try_center (bool): If True, attempt to center the Fermi level in a gap,. Default is False.
            npts (int): Number of points for DOS calculation. Default is 1000.
            curvature (bool): Whether to include curvature in the DOS calculation, used when method is 'refined'. Default is False.

        (2) 'egrid'
            average (bool): If True, average the conduction band minimum and valence band maximum. Default is True.
            npts (int): Number of points for energy grid calculation. Default is 1000.

        Returns:
            float: Fermi energy (eV).
        '''
        self._ensure_interpolation()

        if method == 'vasp':
            fermi = self.data.fermi / units.eV

        elif method == 'solve_mu':
            T = kwargs.get('T', 300)
            refine = kwargs.get('refine', False)
            try_center = kwargs.get('try_center', False)
            npts = kwargs.get('npts', 1000)
            curvature = kwargs.get('curvature', False)
            fermi = self._solve_for_mu(T, refine, try_center, npts, curvature)
        
        elif method == 'egrid':
            average = kwargs.get('average', True)
            npts = kwargs.get('npts', 1000)
            fermi = self._get_fermi_energy_from_egrid(average, npts)

        else:
            raise ValueError(f"Unknown method '{method}' for getting Fermi energy.")
        
        if update:
            self.fermi = fermi
        return fermi
    
    def _solve_for_mu(self, T=300, refine=False, try_center=False, npts=1000, curvature=False):
        '''
        Estimate the chemical potential required to have N0 electrons.
        '''
        self.DOS.calc(npts=npts, curvature=curvature)
        fermi = BL.solve_for_mu(self.DOS.epsilon, self.DOS.dos, self.data.nelect, T, self.data.dosweight, refine=refine, try_center=try_center)
        return fermi / units.eV

    def _get_fermi_energy_from_egrid(self, average=True, npts=1000):
        '''
        Internal method to calculate Fermi energy from energy grid.
        '''
        self.BandStructure.calc(npts)
        egrid = self.BandStructure.egrid
        ivbm = int(self.data.nelect / 2) - 1
        if average:
            fermi = (np.max(egrid[ivbm]) + np.min(egrid[ivbm+1])) / 2.0
        else:
            fermi = np.max(egrid[ivbm])
        return fermi / units.eV

    def plot_dos(self, plot, npts=1000, curvature=False, **kwargs):
        plt.clf()
        plt.cla()
        plt.close()
        self._ensure_interpolation()

        self.DOS.calc(npts=npts, curvature=curvature)
        self.DOS.plot(plot=plot, **kwargs)
        plt.clf()
        plt.cla()
        plt.close()
    def plot_bandstructure(self, bandpath='Auto', npts=1000, **kwargs):
        
        plt.clf()
        plt.cla()
        plt.close()
        self._ensure_interpolation()

        if bandpath != 'Auto':
            self.BandStructure.set_bandpath(bandpath)
        self.BandStructure.calc(npts=npts)
        self.BandStructure.plot(**kwargs)
        plt.clf()
        plt.cla()
        plt.close()

    def plot_band_dos(self, bandpath='Auto', dos_plot='Smoothed', npts=1000, **kwargs):
        plt.clf()
        plt.cla()
        plt.close()
        """
        Plot the interpolated band structure alongside the density of states (DOS).
        
        Kwargs:
            energy_range (tuple): Energy range for the y-axis in the plot, specified as (min_energy, max_energy). Default is (-12, 10).
            figsize (tuple): Figure size specified as (width, height). Default is (9, 6).
            savefig (str): Path to save the figure. If None, the figure is not saved.
            width_ratios (list): Ratios of the widths of the subplots. Default is [2, 1].
        """
        self._ensure_interpolation()

        fermi = self.get_fermi_energy()

        if self.DOS.dos is None:
            self.DOS.calc(npts=npts)
        epsilon = self.DOS.get_smoothed_epsilon()
        dos     = self.DOS.get_smoothed_dos() 

        if bandpath != 'Auto':
            self.BandStructure.set_bandpath(bandpath)
        if self.BandStructure.egrid is None:
            self.BandStructure.calc(npts=npts)
        x = self.BandStructure.get_kpoints_distances()
        X = self.BandStructure.get_special_points()

        fig = plt.figure(figsize=(9,6))
        gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.05)

        ax_band = fig.add_subplot(gs[0])
        ax_dos = fig.add_subplot(gs[1], sharey=ax_band)

        # Plot band structure
        for iband in range(len(self.BandStructure.egrid)):
            ax_band.plot(x, (self.BandStructure.egrid[iband] / units.eV - fermi), color='black', linewidth=1)
        for l in X:
            ax_band.plot([l, l], [-1 / units.eV, 1 / units.eV], color='grey', linestyle='--', linewidth=0.5)

        ax_band.set_xlim(x[0], x[-1])
        ax_band.set_ylim(-12, 10)
        ax_band.set_ylabel(r"$\varepsilon - \varepsilon_F$ [eV]")
        ax_band.set_xticks(X)
        ax_band.set_xticklabels(list(self.BandStructure.klist))

        ax_dos.plot(dos, epsilon - fermi, color='red', linewidth=1)
        ax_dos.set_xlabel('DOS [states/eV]')
        ax_dos.set_xlim(0, max(dos) * 1.1)
        ax_dos.axhline(y=0, color='grey', linestyle='--', linewidth=0.5)
        ax_dos.set_ylim(ax_band.get_ylim())
        ax_dos.set_yticklabels([])

        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()
    @staticmethod
    def _handle_plot_kwargs(figpath, **kwargs):
        '''
        Handle common plotting kwargs and save the figure if requested.

        Kwargs:
            figsize (tuple): Size of the figure (width, height).
            save (bool): Whether to save the figure.
            path (str): Path to save the figure.
            transparent (bool): Whether the background of the figure is transparent.
        '''
        figsize = kwargs.get('figsize', (6,4))
        save = kwargs.get('save', False)
        path = kwargs.get('path', figpath)
        dpi = kwargs.get('dpi', 500)
        transparent = kwargs.get('transparent', False)
        
        return figsize, save, path, dpi, transparent

    class _DOS:
        '''
        Initialize the _DOS class with data from the parent btp2 instance.
        '''
        def __init__(self, btp2_instance):
            self.data_path = btp2_instance.data_path
            self.data = btp2_instance.data
            self.equivalences = btp2_instance.equivalences     
            self.coeffs = btp2_instance.coeffs
            self.metadata = btp2_instance.metadata
            self.fermi = self.data.fermi / units.eV
            self.lattvec = self.data.get_lattvec()
            self.curvature = False
            self.epsilon = self.dos = self.vvdos = self.cdos = None
            self.epsilon_vasp = self.dos_vasp = self.idos_vasp = None
            self.epsilon_smooth = self.dos_smooth = None

        def calc(self, npts=1000, curvature=False):
            '''
            Calculate the density of states (DOS) using interpolation data.

            Args:
                npts (int): Number of points of DOS.
                curvature (bool): Whether to include curvature in the calculation. (default: False)
            '''
            eband, vvband, cband = fite.getBTPbands(self.equivalences, self.coeffs, self.lattvec, curvature=curvature)
            epsilon, dos, vvdos, cdos = BL.BTPDOS(eband, vvband, cband=cband if curvature else None, npts=npts)

            self.epsilon = np.array(epsilon) / units.eV  
            self.dos     = np.array(dos) * units.eV   
            self.vvdos   = np.array(vvdos)
            self.cdos    = np.array(cdos)

        def calc_DOSCAR(self):
            '''
            Get the density of states(DOS) directly from VASP DOSCAR file.
            '''
            filename = os.path.join(self.data_path, 'DOSCAR')
            with open(filename, 'r') as file:
                lines = file.readlines()
            dos_data = lines[6:]

            energies, total_dos, integrated_dos = [], [], []

            for line in dos_data:
                values = line.split()
                if len(values) == 3:  
                    energies.append(float(values[0]))
                    total_dos.append(float(values[1]))
                    integrated_dos.append(float(values[2]))

            self.epsilon_vasp  = np.array(energies)
            self.dos_vasp      = np.array(total_dos)
            self.idos_vasp     = np.array(integrated_dos)

        def smooth(self, window_size=10):
            '''
            Smooth the density of states (DOS) using a moving average.

            Args:
                window_size (int): Size of the window for the moving average.
            '''
            if self.dos is None:
                raise ValueError('DOS must be calculated before smoothing')

            smoothed_dos = np.convolve(self.dos, np.ones(window_size)/window_size, mode='valid')
            adjusted_epsilon = self.epsilon[(window_size-1)//2 : -(window_size//2)]

            self.epsilon_smooth = adjusted_epsilon
            self.dos_smooth     = smoothed_dos

        def get_epsilon(self):
            if self.epsilon is None:
                self.calc()
            return self.epsilon
        
        def get_dos(self):
            if self.dos is None:
                self.calc()
            return self.dos
        
        def get_vvdos(self):
            if self.vvdos is None:
                self.calc()
            return self.vvdos
        
        def get_cdos(self):
            if self.cdos is None:
                if self.curvature == False:
                    print('Calculation done with no curvature.')
                else:
                    self.calc()
            return self.cdos
        
        def get_epsilon_vasp(self):
            if self.epsilon_vasp is None:
                self.calc_DOSCAR()
            return self.epsilon_vasp
        
        def get_dos_vasp(self):
            if self.dos_vasp is None:
                self.calc_DOSCAR()
            return self.dos_vasp
        
        def get_integrated_dos_vasp(self):
            if self.idos_vasp is None:
                self.calc_DOSCAR()
            return self.idos_vasp
        
        def get_smoothed_epsilon(self):
            if self.epsilon_smooth is None:
                if self.epsilon is None:
                    self.calc()
                self.smooth()
            return self.epsilon_smooth
        
        def get_smoothed_dos(self):
            if self.dos_smooth is None:
                if self.dos is None:
                    self.calc()
                self.smooth()
            return self.dos_smooth

        def plot(self, plot = "VASP", **kwargs):
            plt.clf()
            plt.cla()
            plt.close()
            '''
            Plot the density of states.

            Args:
                plot_elements (list): List of elements to plot. Options are ['Boltztrap', 'Smoothed', 'VASP'].
            
            Kwargs:
                fermi (float): Fermi energy (eV). Use to set the Fermi energy manually.
            '''
            fermi = kwargs.get('fermi', self.fermi)
            figpath = './DOS.png'
            figsize, save, path, dpi, transparent = btp2._handle_plot_kwargs(figpath, **kwargs)

            plt.figure(figsize=figsize)

            epsilon = None
            dos = None
            if 'Boltztrap' in plot:
                epsilon = self.get_epsilon()
                dos     = self.get_dos()
                plt.plot(epsilon - fermi, dos, label='Boltztrap')
            
            if 'Smoothed' in plot:
                epsilon = self.get_smoothed_epsilon()
                dos     = self.get_smoothed_dos()                
                plt.plot(epsilon - fermi, dos, label='Smoothed')

            if 'VASP' in plot:
                epsilon = self.get_epsilon_vasp()
                dos     = self.get_dos_vasp()
                plt.plot(epsilon - fermi, dos, label='VASP')
  
            plt.xlabel(r"$\varepsilon - \varepsilon_F$ [eV]")
            plt.ylabel('Density of States (states/eV)')
            if len(plot) > 1:
                plt.legend()
            if save:
                plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')
            else:
                plt.show()
            plt.clf()
            plt.cla()
            plt.close()
            pltout = epsilon-fermi
            return pltout, dos

    class _BandStructure:
        def __init__(self, btp2_instance):
            self.data_path    = btp2_instance.data_path
            self.data         = btp2_instance.data
            self.equivalences = btp2_instance.equivalences     
            self.coeffs       = btp2_instance.coeffs
            self.metadata     = btp2_instance.metadata
            self.fermi        = self.data.fermi / units.eV
            self.lattvec      = self.data.get_lattvec()
            self.klist        = None
            self.kpts_distances = None
            self.special_points = None
            self.egrid        = None
            self.vgrid        = None

        def get_bandpath_vasp(self):
            '''
            Get the band path from the VASP OUTCAR file
            '''
            slab   = read(os.path.join(self.data_path, 'OUTCAR'))
            lat    = slab.cell.get_bravais_lattice()
            points = lat.get_special_points()
            klist  = ''.join(points.keys())

            return klist

        def get_bandpath(self):
            '''
            Get the band path.
            '''
            if self.klist is None:
                print('Automatically finding the band path from VASP OUTCAR.')
                klist = self.get_bandpath_vasp()
                self.klist = klist
            return self.klist
        
        def set_bandpath(self, bandpath):
            '''
            Set the band path

            Args:
                bandpath (str): K points path (ex.'GXWLGK').
            '''
            self.klist = bandpath

        def calc(self, npts=1000):
            '''
            Get the band structure.

            Args:
                npts  : Number of points to calculate
            '''
            klist = self.get_bandpath()

            slab = read(os.path.join(self.data_path, 'OUTCAR'))
            path = bandpath(path=klist, cell=slab.cell, npoints=npts)
            kpts = path.kpts
            x, X, labels = path.get_linear_kpoint_axis()
            egrid, vgrid = fite.getBands(kpts, self.equivalences, self.lattvec, self.coeffs)
        
            self.kpts_distances = x
            self.special_points = X
            self.egrid  = egrid
            self.vgrid  = vgrid

        def get_kpoints_distances(self):
            if self.kpts_distances is None:
                self.calc()
            return self.kpts_distances
        
        def get_special_points(self):
            if self.special_points is None:
                self.calc()
            return self.special_points

        def plot(self, **kwargs):
            plt.clf()
            plt.cla()
            plt.close()
            '''
            Plot the band structure.
            
            Kwargs:
                savefig (str): Path to save the figure 
            '''
            figpath = './bandstructure.png'
            figsize, save, path, dpi, transparent = btp2._handle_plot_kwargs(figpath, **kwargs)
            plt.figure(figsize=figsize)

            x = self.get_kpoints_distances()
            X = self.get_special_points()
            
            for iband in range(len(self.egrid)):
                plt.plot(x, (self.egrid[iband] / units.eV - self.fermi), "k-")
            for l in X:
                plt.plot([l, l], [-1 / units.eV , 1 / units.eV], "k-")

            plt.xlim([x[0], x[-1]])
            plt.ylim(-12, 10)
            plt.ylabel(r"$\varepsilon - \varepsilon_F$ [eV]")
            plt.xticks(ticks=self.special_points, labels=list(self.klist))
            if save:
                plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')
                print('Plot saved as', path)
            else:
                plt.show()
                
            plt.clf()
            plt.cla()
            plt.close()
        
    class _FermiIntegrals:
        def __init__(self, btp2_instance):
            self.data_path    = btp2_instance.data_path
            self.data         = btp2_instance.data
            self.equivalences = btp2_instance.equivalences     
            self.coeffs       = btp2_instance.coeffs
            self.metadata     = btp2_instance.metadata
            self.fermi        = self.data.fermi / units.eV
            self.lattvec      = self.data.get_lattvec()
            self.tmin         = 300
            self.tmax         = 1000
            self.tsteps       = 100
            self.curvature    = True
            self.sigma        = None
            self.seebeck      = None
            self.kappa        = None
            self.Hall         = None

        def get_temp_min(self):
            return self.tmin
        
        def set_temp_min(self, temp):
            self.tmin = temp

        def get_temp_max(self):
            return self.tmax
        
        def set_temp_max(self, temp):
            self.tmax = temp
        
        def get_temp_steps(self):
            return self.tsteps
        
        def set_temp_steps(self, steps):
            self.tsteps = steps

        def set_temp(self, tmin, tmax, tsteps):
            self.tmin   = tmin
            self.tmax   = tmax
            self.tsteps = tsteps

        def _print_progress(self, start_time):
            """
            Print the elapsed time every second.
            """
            while self._is_calculating:
                time.sleep(1)
                elapsed_time = datetime.now() - start_time
                print(f"\rCalculating... Elapsed time: {str(elapsed_time).split('.')[0]}", end='')
            elapsed_time = datetime.now() - start_time
            print(f"\rCalculation finished. Total elapsed time: {str(elapsed_time).split('.')[0]}")

        def calc(self, **kwargs):
            """
            Calculate transport coefficients.

            Args:
                npts (int): Number of points of DOS. (default: 1000)
                mu (list): Range of chemical potential (eV).
                tmin (float): Minimum temperature (K). (default: 300)
                tmax (float): Maximum temperature (K). (default: 1000)
                tsteps (int): Number of temperature steps. (default: 100)
                curvature (bool): Whether to include curvature in the calculation. (default: True)
            
            Returns:
                Updates sigma, seebeck, kappa, Hall with calculated values.
            """
            self.npts = kwargs.get('npts', 1000)
            self.tmin = kwargs.get('tmin', self.tmin)
            self.tmax = kwargs.get('tmax', self.tmax)
            self.tsteps = kwargs.get('tsteps', self.tsteps)
            curvature = kwargs.get('curvature', self.curvature)

            print(f"Calculating at tmin={self.tmin}, tmax={self.tmax}, tsteps={self.tsteps}, curvature={curvature}")
            start_time = datetime.now()
            self._is_calculating = True
            progress_thread = threading.Thread(target=self._print_progress, args=(start_time,))
            progress_thread.start()

            eband, vvband, cband = fite.getBTPbands(self.equivalences, self.coeffs, self.lattvec, curvature=curvature)
            epsilon, dos, vvdos, cdos = BL.BTPDOS(eband, vvband, cband=cband, npts=self.npts)
            fermi = BL.solve_for_mu(epsilon=epsilon, dos=dos, N0=self.data.nelect, T=300, dosweight=self.data.dosweight)
            self.fermi = fermi / units.eV
            
            mu = kwargs.get('mu', None)

            if mu is None:
                mur = epsilon[100:-100]  # [Hartree]
            else:
                indices = (epsilon > mu[0]) & (epsilon < mu[1])
                mur = epsilon[indices]
            TEMP = np.linspace(self.tmin, self.tmax, self.tsteps)

            N, L0, L1, L2, Lm11 = BL.fermiintegrals(epsilon, dos, vvdos, mur=mur, Tr=TEMP, dosweight=self.data.dosweight, cdos=cdos)
            UCvol = self.data.get_volume()
            sigma, seebeck, kappa, Hall = BL.calc_Onsager_coefficients(L0, L1, L2, mur, TEMP, UCvol, Lm11=Lm11)

            self.sigma   = sigma
            self.seebeck = seebeck
            self.kappa   = kappa
            self.Hall    = Hall
            self.mu      = mur / units.eV  # unit conversion from [Hartree] to [eV]

            self._is_calculating = False
            progress_thread.join()

        def _needs_recalculation(self, **kwargs):
            """
            Check if recalculation is needed based on the kwargs.
            """
            if 'tmin' in kwargs and kwargs['tmin'] != self.tmin:
                return True
            if 'tmax' in kwargs and kwargs['tmax'] != self.tmax:
                return True
            if 'tsteps' in kwargs and kwargs['tsteps'] != self.tsteps:
                return True
            if 'curvature' in kwargs and kwargs['curvature'] != self.curvature:
                return True
            return False

        def get_sigma(self, **kwargs):
            """
            Calculate electrical conductivity.

            Args:
                npts (int): Number of points of DOS. (default: 1000)
                mu (list): Range of chemical potential (eV).
                tmin (float): Minimum temperature (K). (default: 300)
                tmax (float): Maximum temperature (K). (default: 1000)
                tsteps (int): Number of temperature steps. (default: 100)
                curvature (bool): Whether to include curvature in the calculation. (default: True)
            
            Returns:
                numpy.ndarray: Electrical conductivity.
            """
            recalculate = False

            if self.sigma is None or self._needs_recalculation(**kwargs):
                self.calc(**kwargs)
            return self.sigma
        
        def get_seebeck(self, **kwargs):
            """
            Calculate Seebeck coefficient.

            Args:
                npts (int): Number of points of DOS. (default: 1000)
                mu (list): Range of chemical potential (eV).
                tmin (float): Minimum temperature (K). (default: 300)
                tmax (float): Maximum temperature (K). (default: 1000)
                tsteps (int): Number of temperature steps. (default: 100)
                curvature (bool): Whether to include curvature in the calculation. (default: True)
            
            Returns:
                numpy.ndarray: Seebeck coefficient.
            """
            if self.seebeck is None or self._needs_recalculation(**kwargs):
                self.calc(**kwargs)
            return self.seebeck
        
        def get_kappa(self, **kwargs):
            """
            Calculate thermal conductivity.

            Args:
                npts (int): Number of points of DOS. (default: 1000)
                mu (list): Range of chemical potential (eV).
                tmin (float): Minimum temperature (K). (default: 300)
                tmax (float): Maximum temperature (K). (default: 1000)
                tsteps (int): Number of temperature steps. (default: 100)
                curvature (bool): Whether to include curvature in the calculation. (default: True)
            
            Returns:
                numpy.ndarray: Thermal conductivity.
            """
            if self.kappa is None or self._needs_recalculation(**kwargs):
                self.calc(**kwargs)
            return self.kappa
        
        def get_Hall(self, **kwargs):
            """
            Calculate Hall coefficient.

            Args:
                npts (int): Number of points of DOS. (default: 1000)
                mu (list): Range of chemical potential (eV).
                tmin (float): Minimum temperature (K). (default: 300)
                tmax (float): Maximum temperature (K). (default: 1000)
                tsteps (int): Number of temperature steps. (default: 100)
                curvature (bool): Whether to include curvature in the calculation. (default: True)
            
            Returns:
                numpy.ndarray: Hall coefficient.
            """
            if self.Hall is None or self._needs_recalculation(**kwargs):
                self.calc(**kwargs)
            return self.Hall
        
        def get_kappa_electron(self):
            """
            Calculate the electric contribution on the thermal conductivity based on Wiedemann-Franz law.

            Returns:
                numpy.ndarray: kappa_electron (nT, nmu, 3, 3)
            """
            lorenz_number = 2.44e-8
            sigma = self.get_sigma()
            temp = np.linspace(self.tmin, self.tmax, self.tsteps)
            temp_ex = temp[:, np.newaxis, np.newaxis, np.newaxis]
            return sigma * temp_ex * lorenz_number
        
        def get_kappa_lattice(self):
            """
            Calculate the lattice contribution on the thermal conductivity based on Seebeck effect.

            Returns:
                numpy.ndarray: kappa_lattice (nT, nmu, 3, 3)
            """
            seebeck = self.get_seebeck()
            sigma = self.get_sigma()
            temp = np.linspace(self.tmin, self.tmax, self.tsteps)
            temp_ex = temp[:, np.newaxis, np.newaxis, np.newaxis]
            return (seebeck ** 2) * sigma * temp_ex
        
        def get_power_factor(self, **kwargs):
            """
            Calculate the power factor.

            Returns:
                numpy.ndarray: Power factor (nT, nmu, 3, 3)
            """
            sigma = self.get_sigma(**kwargs)
            seebeck = self.get_seebeck()

            power_factor = seebeck ** 2 * sigma
            return power_factor
        
        def get_figure_of_merit(self):
            """
            Calculate the figure of merit(zT).

            Returns:
                numpy.ndarray: Figure of merit(zT).
            """
            sigma = self.get_sigma()
            seebeck = self.get_seebeck()
            temp = np.linspace(self.tmin, self.tmax, self.tsteps)
            
            zT = np.zeros_like(seebeck)
            for k in range(seebeck.shape[0]):
                zT[k] = (seebeck[k] ** 2 * sigma[k] * temp[k]) / self.kappa[k]
            return zT

        def get_temperature_array(self):
            return np.linspace(self.tmin, self.tmax, self.tsteps)
        
        def get_chemical_potential_array(self):
            return self.mu - self.fermi 
        
        @staticmethod
        def _scalarize(data):
            return (data[..., 0, 0] + data[..., 1, 1] + data[..., 2, 2]) / 3.0
        
        @staticmethod
        def _get_component(data, direction):
            if direction == 'x':
                return data[..., 0, 0]
            elif direction == 'y':
                return data[..., 1, 1]
            elif direction == 'z':
                return data[..., 2, 2]
            elif direction == 'xy':
                return data[..., 0, 1]
            elif direction == 'xz':
                return data[..., 0, 2]
            elif direction == 'yz':
                return data[..., 1, 2]
            else:
                return btp2._FermiIntegrals._scalarize(data)

        def _get_unit(self, property_name):
            unit_dict = {'sigma'  : r"10$^{14}$ $\mathrm{\Omega^{-1} m^{-1} s^{-1}}$",
                         'seebeck': r"$V K^{-1}",
                         'kappa'  : r"10$^{14}$ $\mathrm{W m}^{-1} \mathrm{K}^{-1}$",
                         'kappa_e': r"10$^{14}$ $\mathrm{W m}^{-1} \mathrm{K}^{-1}$",
                         'kappa_l': r"10$^{14}$ $\mathrm{W m}^{-1} \mathrm{K}^{-1}$",
                         'power factor': r"10$^{14}$ $\mu$W/(cm K$^2$) s",
                         'zT'     : ''}
            return unit_dict[property_name]
        
        def _get_label(self, property_name, component=None):
            label_dict = {
                'sigma': [
                    r"$\sigma/\tau_0$", 
                    r"$\sigma^{(xx)}/\tau_0$", 
                    r"$\sigma^{(yy)}/\tau_0$", 
                    r"$\sigma^{(zz)}/\tau_0$", 
                    r"$\sigma^{(xy)}/\tau_0$", 
                    r"$\sigma^{(xz)}/\tau_0$", 
                    r"$\sigma^{(yz)}/\tau_0$"
                ],
                'seebeck': [
                    r"$S$", 
                    r"$S^{(xx)}$", 
                    r"$S^{(yy)}$", 
                    r"$S^{(zz)}$", 
                    r"$S^{(xy)}$", 
                    r"$S^{(xz)}$", 
                    r"$S^{(yz)}$"
                ],
                'kappa': [
                    r"$\kappa/\tau_0$", 
                    r"$\kappa^{(xx)}/\tau_0$", 
                    r"$\kappa^{(yy)}/\tau_0$", 
                    r"$\kappa^{(zz)}/\tau_0$", 
                    r"$\kappa^{(xy)}/\tau_0$", 
                    r"$\kappa^{(xz)}/\tau_0$", 
                    r"$\kappa^{(yz)}/\tau_0$"
                ],
                'kappa_e': [
                    r"$L\sigma T$", 
                    r"$L\sigma^{(xx)} T$", 
                    r"$L\sigma^{(yy)} T$", 
                    r"$L\sigma^{(zz)} T$", 
                    r"$L\sigma^{(xy)} T$", 
                    r"$L\sigma^{(xz)} T$", 
                    r"$L\sigma^{(yz)} T$"
                ],
                'kappa_l': [
                    r"$S^2\sigma/\tau_0 T$", 
                    r"$S^2\sigma^{(xx)}/\tau_0 T$", 
                    r"$S^2\sigma^{(yy)}/\tau_0 T$", 
                    r"$S^2\sigma^{(zz)}/\tau_0 T$", 
                    r"$S^2\sigma^{(xy)}/\tau_0 T$", 
                    r"$S^2\sigma^{(xz)}/\tau_0 T$", 
                    r"$S^2\sigma^{(yz)}/\tau_0 T$"
                ],
                'power factor': [
                    r"$S^2\sigma/\tau_0$", 
                    r"$S^2\sigma^{(xx)}/\tau_0$", 
                    r"$S^2\sigma^{(yy)}/\tau_0$", 
                    r"$S^2\sigma^{(zz)}/\tau_0$", 
                    r"$S^2\sigma^{(xy)}/\tau_0$", 
                    r"$S^2\sigma^{(xz)}/\tau_0$", 
                    r"$S^2\sigma^{(yz)}/\tau_0$"
                ],
                'zT': [
                    r"$zT$", 
                    r"$zT^{(xx)}$", 
                    r"$zT^{(yy)}$", 
                    r"$zT^{(zz)}$", 
                    r"$zT^{(xy)}$", 
                    r"$zT^{(xz)}$", 
                    r"$zT^{(yz)}$"
                ]}
            if component in ('x', 'xx'):
                idx = 1
            elif component in ('y', 'yy'):
                idx = 2
            elif component in ('z', 'zz'):
                idx = 3
            elif component in ('xy', 'yx'):
                idx = 4
            elif component in ('xz', 'zx'):
                idx = 5
            elif component in ('yz', 'zy'):
                idx = 6
            else:
                idx = 0
            return label_dict[property_name][idx]

        def _get_y_values(self, property_name):
            if property_name == 'sigma':
                return self.get_sigma()
            elif property_name == 'seebeck':
                return self.get_seebeck()
            elif property_name == 'kappa':
                return self.get_kappa()
            elif property_name == 'kappa_e':
                return self.get_kappa_electron()
            elif property_name == 'kappa_l':
                return self.get_kappa_lattice()
            elif property_name == 'power factor':
                return self.get_power_factor()
            elif property_name == 'zT':
                return self.get_figure_of_merit()
            else:
                raise ValueError(f"Invalid property name: {property_name}")
        
        def plot(self, x, y, component=None, temp_range=None, 
                 mu_range=None, fixed_temp=300, fixed_mu=0, **kwargs):
            plt.clf()
            plt.cla()
            plt.close()
            
            """
            General plot function.

            Args:
                x (str): x-axis value, 'temperature' or 'chemical_potential'.
                y (str): y-axis value, 'sigma', 'seebeck', 'kappa', 'powfactor', 'zT'.
                component (str): 'x', 'y', 'z' or None (for scalar).
                temp_range (tuple or list): (min_temp, max_temp) or list of specific temperatures.
                mu_range (tuple or list): (min_mu, max_mu) or list of specific chemical potentials.
                fixed_temp (float or list): Fixed temperature(s) if x_axis is 'chemical_potential'.
                fixed_mu (float or list): Fixed chemical potential(s) if x_axis is 'temperature'.
            """

            y_values_tensor = self._get_y_values(y)
            y_values = self._get_component(y_values_tensor, component)  ## zT, power factor에 대해 반영 필요

            figpath = './{y}_plot.png'
            figsize, save, path, dpi, transparent = btp2._handle_plot_kwargs(figpath, **kwargs)
            realx = None
            realy = None
            if x == 'temperature':
                x_values = self.get_temperature_array()
                if temp_range:
                    temp_start, temp_end = temp_range
                    temp_indices = (x_values >= temp_start) & (x_values <= temp_end)
                    x_values = x_values[temp_indices]
                    y_values = y_values[temp_indices, :]
                if isinstance(fixed_mu, list):
                    for mu in fixed_mu:
                        mu_idx = np.argmin(np.abs(self.get_chemical_potential_array() - mu))
                        plt.plot(x_values, y_values[:, mu_idx], label=f'{mu:.2f} eV')
                else:
                    mu_idx = np.argmin(np.abs(self.get_chemical_potential_array() - fixed_mu))
                    plt.plot(x_values, y_values[:, mu_idx], label=f'{fixed_mu:.2f} eV')
                realx=x_values
                realy=y_values[:, mu_idx]
                xlabel = r"T [K]"

            elif x == 'chemical_potential':
                x_values = self.get_chemical_potential_array()
                if mu_range:
                    mu_start, mu_end = mu_range
                    mu_indices = (x_values >= mu_start) & (x_values <= mu_end)
                    x_values = x_values[mu_indices]
                    y_values = y_values[:, mu_indices]
                if isinstance(fixed_temp, list):
                    for temp in fixed_temp:
                        temp_idx = np.argmin(np.abs(self.get_temperature_array() - temp))
                        plt.plot(x_values, y_values[temp_idx, :], label=f'{temp:.0f} K')
                else:
                    temp_idx = np.argmin(np.abs(self.get_temperature_array() - fixed_temp))
                    plt.plot(x_values, y_values[temp_idx, :], label=f'{fixed_temp:.0f} K')
                
                realx=x_values
                realy=y_values[temp_idx, :]
                xlabel = r"$\mu - \varepsilon_F$ [eV]"

            label = self._get_label(property_name=y, component=component)
            unit = self._get_unit(property_name=y)
            ylabel = label + ' [' + unit + ']'      

            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend()
            if save:
                plt.savefig(path, dpi=dpi, transparent=transparent, bbox_inches='tight')
                print('Plot saved as', path)
            else:
                plt.show()
            plt.clf()
            plt.cla()
            plt.close()
            return realx,realy
            