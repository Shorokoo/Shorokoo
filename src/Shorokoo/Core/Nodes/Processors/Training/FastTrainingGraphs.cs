using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Container for the three computation graphs that define a training setup.
    /// </summary>
    internal class FastTrainingGraphs
    {
        public InternalComputationGraph ModelGraph { get; }
        public InternalComputationGraph LossGraph { get; }
        public InternalComputationGraph OptimizerGraph { get; }

        public FastTrainingGraphs(InternalComputationGraph modelGraph, InternalComputationGraph lossGraph, InternalComputationGraph optimizerGraph)
        {
            ModelGraph = modelGraph;
            LossGraph = lossGraph;
            OptimizerGraph = optimizerGraph;
        }
    }
}
