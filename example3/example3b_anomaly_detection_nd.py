## -------------------------------------------------------------------------------------------------
## -- Paper      : MLPro 2.0 - Online machine learning in Python
## -- Journal    : ScienceDirect, Machine Learning with Applications (MLWA)
## -- Authors    : Detlef Arend, Laxmikant Shrikant Baheti, Steve Yuwono, 
## --              Syamraj Purushamparambil Satheesh Kumar, Andreas Schwung
## -- Module     : example3b_anomaly_detection_nd.py
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2025-04-26)

This module demonstrates the use of anomaly detector based on local outlier factor algorithm with MLPro.
To this regard, a stream of a stream provider is combined with a stream workflow to a stream scenario.
The workflow consists of a standard task 'Aanomaly Detector'.

You will learn:

1) How to set up a stream workflow based on stream tasks.

2) How to set up a stream scenario based on a stream and a processing stream workflow.

3) How to add a task anomalydetector.

4) How to reuse an anomaly detector algorithm from scikitlearn (https://scikit-learn.org/), specifically
Local Outlier Factor

"""

from mlpro.bf.various import Log
from mlpro.bf.ops import Mode
from mlpro.bf.plot import PlotSettings
from mlpro.bf.streams.streams import StreamMLProPOutliers
from mlpro.oa.streams import OAStreamWorkflow, OAStreamScenario

from sklearn.neighbors import LocalOutlierFactor as LOF
from mlpro_int_sklearn.wrappers.anomalydetectors import WrAnomalyDetectorSklearn2MLPro




## -------------------------------------------------------------------------------------------------
## -------------------------------------------------------------------------------------------------
class AdScenario4ADlof (OAStreamScenario):

    C_NAME = 'AdScenario4ADlof'

## -------------------------------------------------------------------------------------------------
    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Get the native stream from MLPro stream provider
        mystream = StreamMLProPOutliers( p_functions = ['sin', 'cos', 'const'],
                                         p_outlier_rate = 0.022,
                                         p_seed = 6,
                                         p_visualize = p_visualize, 
                                         p_logging = p_logging )

        # 2 Creation of a workflow
        workflow = OAStreamWorkflow( p_name = 'Anomaly detection using LOF@scikit-learn',
                                     p_range_max = OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada = p_ada,
                                     p_visualize = p_visualize, 
                                     p_logging = p_logging )

         # 3 Set up and wrap the LOF anomaly detector provided by scikit-learn
        wrapped_lof = WrAnomalyDetectorSklearn2MLPro( p_algo_scikit_learn = LOF( n_neighbors = 3 ),
                                                      p_group_anomaly_det = False, 
                                                      p_delay = 3, 
                                                      p_visualize = p_visualize,
                                                      p_logging = p_logging )

        # 4 Add anomaly detection task to workflow
        workflow.add_task( p_task = wrapped_lof )

        # 5 Return stream and workflow
        return mystream, workflow




# 1 Demo setup

# 1.1 Default values
cycle_limit = 360
logging     = Log.C_LOG_ALL
visualize   = True
step_rate   = 2
  
# 1.2 Welcome message
print('\n\n-----------------------------------------------------------------------------------------')
print('Publication: "MLPro 2.0 - Online machine learning in Python"')
print('Journal    : ScienceDirect, Machine Learning with Applications (MLWA)')
print('Authors    : D. Arend, L.S. Baheti, S. Yuwono, S.P.S. Kumar, A. Schwung')
print('Affiliation: South Westphalia University of Applied Sciences, Germany')
print('Sample     : 3a Anomaly detection (3D)')
print('-----------------------------------------------------------------------------------------\n')


# 2 Instantiate the stream scenario
myscenario = AdScenario4ADlof( p_mode = Mode.C_MODE_SIM, 
                               p_cycle_limit = cycle_limit,
                               p_visualize = visualize,
                               p_logging = logging )


# 3 Reset and run own stream scenario
myscenario.reset()

myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                    p_view_autoselect = False,
                                                    p_step_rate = step_rate ) )

input('\nPlease arrange all windows and press ENTER to start stream processing...')

myscenario.run()

input('Press ENTER to exit...')       