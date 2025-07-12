## -------------------------------------------------------------------------------------------------
## -- Paper      : MLPro 2.0 - Online machine learning in Python
## -- Journal    : ScienceDirect, Machine Learning with Applications (MLWA)
## -- Authors    : Detlef Arend, Laxmikant Shrikant Baheti, Steve Yuwono, 
## --              Syamraj Purushamparambil Satheesh Kumar, Andreas Schwung
## -- Module     : example2a_online_clustering_of_stream_data_2d.py
## -------------------------------------------------------------------------------------------------

"""
Ver. 1.0.0 (2024-12-12)

This example demonstrates online cluster analysis of normalized static 2D random point clouds using the wrapped
River implementation of stream algorithm KMeans. To this regard, the systematics of sub-framework 
MLPro-OA-Streams for online adaptive stream processing is used to implement a scenario consisting of  
a custom workflow and a native benchmark stream.

In particular you will learn:

1. How to set up, run and visualize an online adaptive custom stream processing scenario 

2. How to reuse wrapped River algorithms in own custom stream processing workflows

3. How to reuse native MLPro benchmark streams

4. How to reuse native MLPro online adaptive min-max normalization for data preprocessing

"""

from datetime import datetime

from mlpro.bf.streams.streams import StreamMLProClouds
from mlpro.bf.various import Log
from mlpro.bf.plot import PlotSettings
from mlpro.bf.ops import Mode
from mlpro.oa.streams import OAStreamScenario, OAStreamWorkflow
from mlpro.oa.streams.tasks import BoundaryDetector, NormalizerMinMax

from mlpro_int_river.wrappers.clusteranalyzers import WrRiverKMeans2MLPro



# Prepare a scenario for Static 2D Point Clouds
class Static2DScenario(OAStreamScenario):

    C_NAME = 'Static2DScenario'

    def _setup(self, p_mode, p_ada: bool, p_visualize: bool, p_logging):

        # 1 Get stream from StreamMLProClouds
        stream = StreamMLProClouds( p_num_dim = 2,
                                    p_num_instances = 2000,
                                    p_num_clouds = 5,
                                    p_seed = 1,
                                    p_radii = [100, 150, 200, 250, 300],
                                    p_weights = [2,3,4,5,6],
                                    p_logging = Log.C_LOG_NOTHING )

        # 2 Set up a stream workflow based on a custom stream task

        # 2.1 Creation of a workflow
        workflow = OAStreamWorkflow( p_name = 'Cluster Analysis using KMeans@River',
                                     p_range_max = OAStreamWorkflow.C_RANGE_NONE,
                                     p_ada = p_ada,
                                     p_visualize = p_visualize,
                                     p_logging = p_logging )


        # 2.2 Creation of tasks and add them to the workflow

        # Boundary detector 
        task_bd = BoundaryDetector( p_name = '#1: Boundary Detector', 
                                    p_ada = p_ada, 
                                    p_visualize = p_visualize,   
                                    p_logging = p_logging)
        
        workflow.add_task(p_task = task_bd)

        # MinMax-Normalizer
        task_norm_minmax = NormalizerMinMax( p_name = '#2: Normalizer MinMax', 
                                             p_ada = p_ada,
                                             p_visualize = p_visualize, 
                                             p_logging = p_logging )

        task_bd.register_event_handler( p_event_id=BoundaryDetector.C_EVENT_ADAPTED,
                                        p_event_handler=task_norm_minmax.adapt_on_event )
        
        workflow.add_task( p_task = task_norm_minmax, p_pred_tasks=[task_bd] )

        # Cluster Analyzer
        task_clusterer = WrRiverKMeans2MLPro( p_name = '#3: KMeans@River',
                                              p_n_clusters = 5,
                                              p_halflife = 0.3, 
                                              p_sigma = 0.1, 
                                              p_mu = 0.0,
                                              p_seed = 3, 
                                              p_p = 1,
                                              p_visualize = p_visualize,
                                              p_logging = p_logging )
        
        task_norm_minmax.register_event_handler( p_event_id = NormalizerMinMax.C_EVENT_ADAPTED,
                                                 p_event_handler = task_clusterer.renormalize_on_event )
        
        workflow.add_task( p_task = task_clusterer, p_pred_tasks=[task_norm_minmax] )

        # 3 Return stream and workflow
        return stream, workflow




# 1 Demo setup

# 1.1 Default values
cycle_limit = 500
logging     = Log.C_LOG_ALL
visualize   = True
step_rate   = 1

# 1.2 Welcome message
print('\n\n-----------------------------------------------------------------------------------------')
print('Publication: "MLPro 2.0 - Online machine learning in Python"')
print('Journal    : ScienceDirect, Machine Learning with Applications (MLWA)')
print('Authors    : D. Arend, L.S. Baheti, S. Yuwono, S.P.S. Kumar, A. Schwung')
print('Affiliation: South Westphalia University of Applied Sciences, Germany')
print('Sample     : 2a Online clustering of stream data (2D)')
print('-----------------------------------------------------------------------------------------\n')


# 2 Instantiate the stream scenario
myscenario = Static2DScenario( p_mode = Mode.C_MODE_SIM,
                               p_cycle_limit = cycle_limit,
                               p_visualize = visualize,
                               p_logging=logging )



# 3 Reset and run own stream scenario
myscenario.reset()
myscenario.init_plot( p_plot_settings=PlotSettings( p_view = PlotSettings.C_VIEW_ND,
                                                    p_step_rate = step_rate ) )

input('\nPlease arrange all windows and press ENTER to start stream processing...')

tp_before           = datetime.now()
myscenario.run()
tp_after            = datetime.now()
tp_delta            = tp_after - tp_before
duraction_sec       = ( tp_delta.seconds * 1000000 + tp_delta.microseconds + 1 ) / 1000000
myscenario.log(Log.C_LOG_TYPE_S, 'Duration [sec]:', round(duraction_sec,2), ', Cycles/sec:', round(cycle_limit/duraction_sec,2))

clusters            = myscenario.get_workflow()._tasks[2].clusters
number_of_clusters  = len(clusters)

myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, 'Here is the recap of the cluster analyzer')
myscenario.log(Log.C_LOG_TYPE_I, 'Number of clusters: ', number_of_clusters)
for x in range(number_of_clusters):
    myscenario.log(Log.C_LOG_TYPE_I, 'Center of Cluster ', str(x+1), ': ', list(clusters[x].centroid.value))
    myscenario.log(Log.C_LOG_TYPE_I, 'Size of Cluster ', str(x+1), ': ', clusters[x].size.value)
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')
myscenario.log(Log.C_LOG_TYPE_I, '-------------------------------------------------------')

input('Press ENTER to exit...')