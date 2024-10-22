import numpy as np
import pandas as pd
#import src.simulation.simulation_functions as funcs
import src.utils.functions as funcs
import src.utils.decorators as decos
import src.models as model
from src.simulation import object_solver, vector_solver
from abc import ABC, abstractmethod
from src.config import check_config

class Simulation:
    """
    The simulation class

    Attributes
        attribute: mesh - The mesh object
        attribute: config - The configuration dictionary
        attribute: time_interval - The time interval
        attribute: data - A pandas dataframe with the oil content at each cell at each time step
        attribute: result - A pd.series with the oil content in the marked area at each time step

    Main methods
        method: run - Run the simulation
        method: analyse - Analyse the simulation results

    Abstract methods
        method: _find_cells_in_marked_area - Find the cells in the marked area
        method: generate_oil_distribution - Generate the oil distribution
        method: check_for_in_medias_res - Check whether the simulation starts from a restart file
    """
    @decos.logging_decorator
    def __init__(self, mesh, config):
        assert mesh.__class__ == model.Mesh, f"Expected mesh to be of type Mesh, got {mesh.__class__} instead"
        assert config.__class__ == dict, f"Expected config to be of type dict, got {config.__class__} instead"
        assert 'settings' in config.keys(), "settings key not found in config"
        assert 'geometry' in config.keys(), "geometry key not found in config"
        assert 'IO' in config.keys(), "IO key not found in config"

        self._mesh = mesh
        self._config = check_config(config)

        self._time_interval = np.linspace(config['settings']['tStart'],
                                         config['settings']['tEnd'],
                                         config['settings']['nSteps'])
        
        # Create DataFrame and set 'cell_id' as index
        self._data = self._initialize_data()
        
        assert isinstance(self.data, pd.DataFrame), f"Expected data to be of type pd.DataFrame, got {self.data.__class__} instead"
        assert isinstance(self.time_interval, np.ndarray), f"Expected time_interval to be of type np.ndarray, got {self.time_interval.__class__} instead"


    @abstractmethod
    @decos.logging_decorator
    def _initialize_data(self):
        """
        Initialize the data DataFrame
        :return: A DataFrame with zeros, indexed by cell IDs and columns as time steps
        """

        # Create a DataFrame with zeros, indexed by cell IDs and columns as time steps
        data = pd.DataFrame(0, index=self.cell_ids, columns=self.time_interval)
        data = data.astype(float)  # Ensure all columns are floats
        
        # Set the index name to 'cell_id'
        data.index.name = 'cell_id'
        
        return data
    

    @abstractmethod
    @decos.logging_decorator
    def _check_for_in_medias_res(self):
        """
        Check whether the simulation starts from a restart file
        :return: True if the simulation starts from a restart file
        """
        if self.config['IO']['restartFile'] is not None and self.config['settings']['tStart'] > 0:
            return True
        else:
            return False
    
    @abstractmethod
    @decos.logging_and_timing_decorator
    def _generate_oil_distribution(self):
        """
        Generate the oil distribution
        :return: An array with the initial oil content in each cell
        """

        in_medias_res = self._check_for_in_medias_res()
        if in_medias_res:
            
            restart_log = pd.read_csv(self.config['IO']['restartFile'])

            # set index equal to column 'cell_id'
            restart_log.set_index('cell_id', inplace=True)

            restart_log_times = restart_log.columns

            # we want to find which time in the restart log is closest to the start time
            time_nearest_start = funcs.find_nearest_value_in_array(restart_log_times, 
                                                                   self.config['settings']['tStart'])

            initial_oil_content = restart_log[time_nearest_start]

            assert len(initial_oil_content) == len(self.mesh.cells), "The number of cells in the mesh and the restart file do not match"

            return np.array(initial_oil_content, dtype=np.float64)
    
        else:

            initial_oil_content = np.zeros(len(self.mesh.cells))

            for cell_id in self.mesh.cells.keys():
                initial_oil_content[cell_id] = funcs.initial_distribution(self.mesh.cells[cell_id].centroid)

            return initial_oil_content

    @decos.logging_and_timing_decorator
    def run_object_solve(self):
        """
        Run the simulation using the object solver
        """
        
        self._data[self.time_interval[0]] = self._generate_oil_distribution()

        dt = self.time_interval[1] - self.time_interval[0]

        area_constants = np.array([cell.area_constant for cell in self.mesh.cells.values()])

        for i, time in enumerate(self.time_interval[1:]):
            last_time = self.time_interval[i] # time interval starts from 1, so the enumeration is lagging by 1
            self._data[time] = object_solver(self.mesh, self.data[last_time], dt, area_constants)

    
    @decos.logging_and_timing_decorator
    def run_fast_solve(self):
        """
        Run the simulation using the fast solver
        """

        # generate matrix with dimensions (n_cells, nSteps)

        data_matrix = np.zeros((self.nCells, self.nSteps), dtype=np.float64)

        # set the first column to the initial oil distribution

        data_matrix[:, 0] = self._generate_oil_distribution()

        dt = self.time_interval[1] - self.time_interval[0]

        area_constants = np.array(self.mesh.area_constant_vector, dtype=np.float64)
        dot_matrix = np.array(self.mesh.dot_matrix, dtype=np.float64)
        neighbour_matrix = np.array(self.mesh.neighbour_matrix, dtype=np.int64)

        for current_iter in range(1, self.nSteps):
            last_iter = current_iter - 1
            # get the oil content at the last iteration
            oil_content_last_iter = data_matrix[:, last_iter]
            # get the oil content at the current iteration
            oil_content_current_iter = vector_solver(neighbour_matrix, 
                                                     dot_matrix, 
                                                     oil_content_last_iter, 
                                                     area_constants, 
                                                     dt)
            # set the oil content at the current iteration
            data_matrix[:, current_iter] = oil_content_current_iter

        # set dataframe to matrix values

        self._data.loc[:, self.time_interval] = data_matrix
            

    @abstractmethod
    @decos.logging_decorator
    def _find_cells_in_marked_area(self):
        """
        Find the cells in the marked area
        :return: A list of the cell IDs in the marked area
        """
        borders = self.config['geometry']['borders']

        cells_in_marked_area = []

        x_min = borders[0][0]
        x_max = borders[0][1]
        y_min = borders[1][0]
        y_max = borders[1][1]

        for cell in self.mesh.cells.values():
            centroid = cell.centroid
            if x_min <= centroid[0] <= x_max and y_min <= centroid[1] <= y_max:
                cells_in_marked_area.append(cell.id)

        return cells_in_marked_area
    
    
    @decos.logging_and_timing_decorator
    def analyse(self):
        """
        Analyse the simulation results
        :return: pd.series with the oil content in the marked area at each time step
        """
        cells_in_marked_area = self._find_cells_in_marked_area()
        
        # Ensure the main DataFrame is indexed by cell_id
        marked_dataframe = self.data.loc[cells_in_marked_area]
        
        # sum over each column and turn into series object
        self._result = marked_dataframe.sum()

        return self._result
    
    @property
    def mesh(self):
        return self._mesh
    
    @property
    def config(self):
        return self._config
    
    @property
    def time_interval(self):
        return self._time_interval

    @property
    def cell_ids(self):
        return self.mesh.cells.keys()
    
    @property
    def nCells(self):
        return len(self.mesh.cells)
    
    @property
    def nSteps(self):
        return self.config['settings']['nSteps']
    
    @property
    def data(self):
        return self._data
    
    @property
    def result(self):
        return self._result