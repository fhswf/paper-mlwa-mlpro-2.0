## -------------------------------------------------------------------------------------------------
## -- Paper      : MLPro 2.0 - Online machine learning in Python
## -- Journal    : ScienceDirect, Machine Learning with Applications (MLWA)
## -- Authors    : Detlef Arend, Laxmikant Shrikant Baheti, Steve Yuwono, 
## --              Syamraj Purushamparambil Satheesh Kumar, Andreas Schwung
## -- Module     : TEP browser.py
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-05-26)

"""

import os

from mlpro.bf.streams.streams import StreamMLProCSV
from mlpro.bf.streams.tasks import *

from mlpro.oa.streams import *



from datetime import datetime

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.math.properties import Properties
from mlpro.bf.math.geometry import cprop_crosshair
from mlpro.bf.streams import *
from mlpro.bf.streams.streams import StreamMLProClusterGenerator
from mlpro.bf.streams.tasks import RingBuffer
from mlpro.oa.streams import OAStreamTask, OAStreamWorkflow, OAStreamScenario
from mlpro.oa.streams.tasks import BoundaryDetector, Normalizer, NormalizerMinMax

from mlpro_int_openml import WrStreamProviderOpenML



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class MovingAverage (OAStreamTask, Properties):
    """
    Sample implementation of an online-adaptive stream task that buffers internal data relevant for
    a renormalization whenever a prio normalizer changes it's parameters. Here, the moving average
    of the incoming instances is calculated and stored. 
    """

    C_NAME              = 'Moving average'

    C_PROPERTIES        = [ cprop_crosshair ]   

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_name = None, 
                  p_range_max = StreamTask.C_RANGE_THREAD, 
                  p_ada : bool = True, 
                  p_buffer_size : int = 0, 
                  p_duplicate_data : bool = False, 
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL, 
                  p_remove_obs : bool = True,
                  **p_kwargs ):
        
        Properties.__init__( self, p_visualize = p_visualize )
       
        OAStreamTask.__init__( self, 
                               p_name = p_name, 
                               p_range_max = p_range_max, 
                               p_ada = p_ada, 
                               p_buffer_size = p_buffer_size, 
                               p_duplicate_data = p_duplicate_data, 
                               p_visualize = p_visualize, 
                               p_logging = p_logging, 
                               **p_kwargs )
                 
        self._moving_avg     = None
        self._num_inst       = 0
        self._remove_obs     = p_remove_obs
        self.crosshair.color = 'red'


## -------------------------------------------------------------------------------------------------
    def _run(self, p_inst):

        # 0 Intro
        inst_avg_id     = -1
        inst_avg_tstamp = None

        
        # 1 Process all incoming new/obsolete stream instances
        for inst_id, (inst_type, inst) in p_inst.items():

            feature_data = inst.get_feature_data().get_values()

            if inst_type == InstTypeNew:
                if self._moving_avg is None:
                    self._moving_avg = feature_data.copy() 
                else:
                    self._moving_avg = ( self._moving_avg * self._num_inst + feature_data ) / ( self._num_inst + 1 )

                self._num_inst += 1

            elif ( inst_type == InstTypeDel ) and self._remove_obs:
                self._moving_avg = ( self._moving_avg * self._num_inst - feature_data ) / ( self._num_inst - 1 )
                self._num_inst  -= 1

            if inst_id > inst_avg_id:
                inst_avg_id     = inst_id
                inst_avg_tstamp = inst.tstamp
                feature_set     = inst.get_feature_data().get_related_set()

        if inst_avg_id == -1: return

            
        # 2 Clear all incoming stream instances
        p_inst.clear()


        # 3 Add a new stream instance containing the moving average 
        inst_avg_data       = Element( p_set = feature_set )
        inst_avg_data.set_values( p_values = self._moving_avg.copy() )
        inst_avg            = Instance( p_feature_data = inst_avg_data, p_tstamp = inst_avg_tstamp )
        inst_avg.id         = inst_avg_id

        p_inst[inst_avg.id] = ( InstTypeNew, inst_avg )

        self.crosshair.value = self._moving_avg
 

## -------------------------------------------------------------------------------------------------
    def _renormalize(self, p_normalizer: Normalizer):
        try:
            self._moving_avg = p_normalizer.renormalize( p_data = self._moving_avg.copy() )
            self.log(Log.C_LOG_TYPE_W, 'Moving avg renormalized')
        except:
            pass


## -------------------------------------------------------------------------------------------------
    def init_plot(self, p_figure = None, p_plot_settings = None):
        OAStreamTask.init_plot( self, p_figure = p_figure, p_plot_settings = p_plot_settings )
        Properties.init_plot( self, p_figure = p_figure, p_plot_settings = p_plot_settings )


## -------------------------------------------------------------------------------------------------
    def update_plot(self, p_inst = None, **p_kwargs):
        OAStreamTask.update_plot( self, p_inst = p_inst, **p_kwargs )
        Properties.update_plot( self, p_inst = p_inst, **p_kwargs )


## -------------------------------------------------------------------------------------------------
    def remove_plot(self, p_refresh = True):
        OAStreamTask.remove_plot(self, p_refresh)
        Properties.remove_plot(self, p_refresh)


## -------------------------------------------------------------------------------------------------
    def _finalize_plot_view(self, p_inst_ref):
        ps_old = self.get_plot_settings().copy()
        OAStreamTask._finalize_plot_view(self,p_inst_ref)
        ps_new = self.get_plot_settings()

        if ps_new.view != ps_old.view:
            self.crosshair._plot_initialized = False
            Properties.init_plot( self, p_figure = self._figure, p_plot_settings = ps_new )
 



## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class DemoScenario (OAStreamScenario):

    C_NAME = 'Auto-renormalization MinMax'

## -------------------------------------------------------------------------------------------------
    def __init__( self, 
                  p_mode = Mode.C_MODE_SIM, 
                  p_ada : bool = True, 
                  p_cycle_limit : int = 0, 
                  p_num_features : int = 2,
                  p_num_inst : int = 1000,
                  p_visualize : bool = False, 
                  p_logging = Log.C_LOG_ALL ):
        
        self._num_features  = p_num_features
        self._num_inst      = p_num_inst

        super().__init__( p_mode = p_mode, 
                          p_ada = p_ada, 
                          p_cycle_limit = p_cycle_limit, 
                          p_visualize = p_visualize, 
                          p_logging = p_logging )


## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Extract dataset 1477 'gas-drift-different-concentrations' from OpenML

        # 1.1 Create a Wrapper for OpenML stream provider
        openml = WrStreamProviderOpenML(p_logging = logging)

        # 1.2 Get stream 'gas-drift-different-concentrations' from the stream provider OpenML
        stream = openml.get_stream( p_name='gas-drift-different-concentrations', p_logging=logging)


        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAStreamWorkflow( p_name = 'Input signal: OpenML 1477',
                                     p_range_max = OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada = p_ada,
                                     p_visualize = p_visualize, 
                                     p_logging = p_logging )

     
        # # 2.2 Add a basic sliding window to buffer some data
        # task_window = RingBuffer( p_buffer_size = 50, 
        #                           p_delay = True,
        #                           p_enable_statistics = True,
        #                           p_name = 'T1 - Sliding window',
        #                           p_duplicate_data = True,
        #                           p_visualize = p_visualize,
        #                           p_logging = p_logging )
        
        # workflow.add_task( p_task = task_window )


        # 2.3 Add a boundary detector and connect to the ring buffer
        task_bd = BoundaryDetector( p_name = 'T2 - Boundary detector', 
                                    p_ada = p_ada, 
                                    p_visualize = p_visualize,
                                    p_logging = p_logging )

        #task_window.register_event_handler( p_event_id = RingBuffer.C_EVENT_DATA_REMOVED, p_event_handler = task_bd.adapt_on_event )
        workflow.add_task( p_task = task_bd )#, p_pred_tasks = [task_window] )


        # # 2.4 Add a MinMax-Normalizer and connect to the boundary detector
        # task_norm_minmax = NormalizerMinMax( p_name = 'T3 - MinMax normalizer', 
        #                                      p_ada = p_ada, 
        #                                      p_duplicate_data = True,
        #                                      p_visualize = p_visualize, 
        #                                      p_logging = p_logging )

        # task_bd.register_event_handler( p_event_id = BoundaryDetector.C_EVENT_ADAPTED, p_event_handler = task_norm_minmax.adapt_on_event )
        # workflow.add_task( p_task = task_norm_minmax, p_pred_tasks = [task_bd] )


        # # 2.5 Add a moving average task for raw data behind the sliding window
        # task_ma = MovingAverage( p_name = 'T4 - Moving average (renormalized)', 
        #                          p_ada = p_ada,
        #                          p_visualize = p_visualize,
        #                          p_logging = p_logging,
        #                          p_centroid_crosshair_labels = False )
        
        # workflow.add_task( p_task = task_ma, p_pred_tasks = [ task_norm_minmax ] )
        # task_norm_minmax.register_event_handler( p_event_id = NormalizerMinMax.C_EVENT_ADAPTED, p_event_handler = task_ma.renormalize_on_event )


        # 3 Return stream and workflow
        return stream, workflow





# 1 Demo setup

# 1.1 Default values
cycle_limit  = 100
logging      = Log.C_LOG_ALL
visualize    = True
step_rate    = 1
plot_horizon = cycle_limit
 
# 1.2 Welcome message
print('\n\n-----------------------------------------------------------------------------------------')
print('Publication: "MLPro 2.0 - Online machine learning in Python"')
print('Journal    : ScienceDirect, Machine Learning with Applications (MLWA)')
print('Authors    : D. Arend, L.S. Baheti, S. Yuwono, S.P.S. Kumar, A. Schwung')
print('Affiliation: South Westphalia University of Applied Sciences, Germany')
print('Sample     : 1a Auto-renormalization of drifting stream data (MinMax)')
print('-----------------------------------------------------------------------------------------\n')


# 2 Instantiate the stream scenario
myscenario = DemoScenario( p_mode=Mode.C_MODE_SIM,
                           p_cycle_limit=cycle_limit,
                           p_visualize=visualize,
                           p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()
myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                    p_view_autoselect = True,
                                                    p_step_rate = step_rate,
                                                    p_plot_horizon = plot_horizon ) )

input('\n\nPlease arrange all windows and press ENTER to start stream processing...')

myscenario.run()

input('Press ENTER to exit...')