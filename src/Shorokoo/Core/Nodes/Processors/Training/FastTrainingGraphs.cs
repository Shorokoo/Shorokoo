using Shorokoo.Graph;

namespace Shorokoo.Core.Nodes.Processors.Training
{
    /// <summary>
    /// Container for the three computation graphs that define a training setup.
    /// </summary>
    internal class FastTrainingGraphs
    {
        public FastComputationGraph ModelGraph { get; }
        public FastComputationGraph LossGraph { get; }
        public FastComputationGraph OptimizerGraph { get; }

        public FastTrainingGraphs(FastComputationGraph modelGraph, FastComputationGraph lossGraph, FastComputationGraph optimizerGraph)
        {
            ModelGraph = modelGraph;
            LossGraph = lossGraph;
            OptimizerGraph = optimizerGraph;
        }
    }
}
