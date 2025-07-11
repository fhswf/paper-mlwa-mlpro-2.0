## -------------------------------------------------------------------------------------------------
## -- Paper      : MLPro 2.0 - Online machine learning in Python
## -- Journal    : ScienceDirect, Machine Learning with Applications (MLWA)
## -- Authors    : Detlef Arend, Laxmikant Shrikant Baheti, Steve Yuwono, 
## --              Syamraj Purushamparambil Satheesh Kumar, Andreas Schwung
## -- Module     : example1b_auto_renormalization_ztrans.py
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-07-07)

This experiment demonstrates how to
- access and process data sets from OpenML using MLPro's integration package mlpro_int_openml
- set up a stream workflow consisting of numerous stream tasks
- configure MLPro's auto-renormalization mechanism using Z-transformation

"""

from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.bf.streams.tasks import Rearranger, RingBuffer

from mlpro.oa.streams import OAStreamTask, OAStreamWorkflow, OAStreamScenario, OAStreamAdaptationType
from mlpro.oa.streams.tasks import NormalizerZTransform, MovingAverage
from mlpro.oa.streams.helpers import OAObserver

from mlpro_int_openml import WrStreamProviderOpenML




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

        # 1 Load the OpenML data stream 1477 'gas-drift'
        stream        = WrStreamProviderOpenML( p_logging = p_logging ).get_stream( p_id = '1477' )
        feature_space = stream.get_feature_space()
        features      = feature_space.get_dims()
        features_new  = [ features[i] for i in [4,5,6] ]  # Features V5, V6,V7
        

        # 2 Set up the stream workflow 
        workflow = OAStreamWorkflow( p_name = 'Input signal: OpenML 1477 "gas-drift"',
                                     p_range_max = OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada = p_ada,
                                     p_visualize = p_visualize, 
                                     p_logging = p_logging )
        

        # 2.1 Add a rearranger to select the features of interest
        task1_rearranger = Rearranger( p_name = 'T1/T2 - Feature extraction and sliding window',
                                       p_visualize = p_visualize,
                                       p_logging = p_logging,
                                       p_features_new = [ ( 'F', features_new ) ] )
        
        workflow.add_task( p_task = task1_rearranger )
        
    
        # 2.2 Add a basic sliding window to buffer some data
        task2_window = RingBuffer( p_buffer_size = 50, 
                                   p_delay = True,
                                   p_enable_statistics = False,
                                   p_name = 'T2 - Sliding window',
                                   p_duplicate_data = True,
                                   p_visualize = p_visualize,
                                   p_logging = p_logging )
        
        workflow.add_task( p_task = task2_window, p_pred_tasks = [ task1_rearranger ] )


        # 2.3 Add a neutral task for the buffered raw data
        task3_raw_buffered = OAStreamTask( p_name = 'T3 - Raw, buffered', 
                                           p_ada = p_ada, 
                                           p_visualize = p_visualize,
                                           p_logging = p_logging )
        
        workflow.add_task( p_task = task3_raw_buffered, p_pred_tasks = [ task2_window ] )


        # 2.4 Add a moving average task for the buffered raw data
        task4_ma_raw = MovingAverage( p_name = 'T4 - Moving average (buffered)', 
                                      p_ada = p_ada,
                                      p_visualize = p_visualize,
                                      p_logging = p_logging,
                                      p_remove_obs = True, 
                                      p_renormalize_plot_data = True )
        
        workflow.add_task( p_task = task4_ma_raw, p_pred_tasks = [ task3_raw_buffered ] )


        # 2.5 Add a ztrans normalizer and connect to the boundary detector
        task5_norm_ztrans = NormalizerZTransform( p_name = 'T5 - ZTrans normalizer', 
                                                  p_ada = p_ada, 
                                                  p_duplicate_data = True,
                                                  p_visualize = p_visualize, 
                                                  p_logging = p_logging )

        workflow.add_task( p_task = task5_norm_ztrans, p_pred_tasks = [task3_raw_buffered] )


        # 2.6 Add a moving average task for raw data behind the normalizer
        task6_ma_renorm = MovingAverage( p_name = 'T6 - Moving average (renormalized)', 
                                         p_ada = p_ada,
                                         p_visualize = p_visualize,
                                         p_logging = p_logging,
                                         p_remove_obs = True, 
                                         p_renormalize_plot_data = True )
        
        workflow.add_task( p_task = task6_ma_renorm, p_pred_tasks = [ task5_norm_ztrans ] )
        task5_norm_ztrans.register_event_handler( p_event_id = NormalizerZTransform.C_EVENT_ADAPTED, p_event_handler = task6_ma_renorm.renormalize_on_event )


        # 3 Additional helpers for online diagnostics

        # 3.1 Observer for online adaptations of the ztrans normalizer
        workflow.add_helper( p_helper = OAObserver( p_related_task = task5_norm_ztrans,
                                                    p_no_per_task = 1,
                                                    p_logarithmic_plot = False,
                                                    p_filter_subtypes = [ OAStreamAdaptationType.FORWARD ],
                                                    p_visualize = p_visualize,
                                                    p_logging = p_logging ) )
        
        workflow.add_helper( p_helper = OAObserver( p_related_task = task5_norm_ztrans,
                                                    p_no_per_task = 2,
                                                    p_logarithmic_plot = False,
                                                    p_filter_subtypes = [ OAStreamAdaptationType.REVERSE ],
                                                    p_visualize = p_visualize,
                                                    p_logging = p_logging ) )
        
        # 3.2 Observer for online adaptations of the moving average task
        workflow.add_helper( p_helper = OAObserver( p_related_task = task6_ma_renorm,
                                                    p_logarithmic_plot = False,
                                                    p_visualize = p_visualize,
                                                    p_logging = p_logging ) )


        # 4 Return stream and workflow
        return stream, workflow




# 1 Demo setup

# 1.1 Default values
time_index_start    = 1300
time_index_stop     = 1400
logging             = Log.C_LOG_ALL
visualize           = True
step_rate           = 1

 
# 1.2 Welcome message
print('\n\n-----------------------------------------------------------------------------------------')
print('Publication: "MLPro 2.0 - Online machine learning in Python"')
print('Journal    : ScienceDirect, Machine Learning with Applications (MLWA)')
print('Authors    : D. Arend, L.S. Baheti, S. Yuwono, S.P.S. Kumar, A. Schwung')
print('Affiliation: South Westphalia University of Applied Sciences, Germany')
print('Sample     : 1b Auto-renormalization of drifting stream data (Z-transformation)')
print('-----------------------------------------------------------------------------------------\n\n')

# 1.3 User input and derived values
i = input(f'Start time index (press ENTER for {time_index_start}): ')
if i != '': time_index_start = int(i)
i = input(f'End time index (press ENTER for {time_index_stop}): ')
if i != '': time_index_stop = int(i)

if time_index_start >= time_index_stop:
    print('\nERROR: Start time index must be less than end time index')
    exit(1)

i = input(f'Visualization step rate (press ENTER for {step_rate}): ')
if i != '': step_rate = int(i)

cycle_limit  = time_index_stop - time_index_start
plot_horizon = cycle_limit
data_horizon = 0

# 2 Instantiate the stream scenario
myscenario = DemoScenario( p_mode=Mode.C_MODE_SIM,
                           p_cycle_limit=cycle_limit,
                           p_visualize=visualize,
                           p_logging=logging )


# 3 Reset and run own stream scenario
myscenario.reset()

# 3.1 Fast forward to start time index
stream_iterator = myscenario._iterator
for ti in range(time_index_start): inst = next(stream_iterator)

myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                    p_view_autoselect = False,
                                                    p_step_rate = step_rate,
                                                    p_plot_horizon = plot_horizon,
                                                    p_data_horizon = data_horizon ) )

input('\n\nPlease arrange all windows and press ENTER to start stream processing...')

myscenario.run()

input('Press ENTER to exit...')