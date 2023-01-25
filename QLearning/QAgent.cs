using System;
using System.Threading.Tasks;

namespace QLearning
{
    public class QAgent
    {
        public double[,] QTable { get; set; }
        public double[,] Traces { get; set; }
        public QAlgorithm Algorithm { get; set; }
        public TraceType TraceType { get; set; }
        public Random Random { get; set; }

        public double Gamma { get; set; } = 0.9;
        public double Alpha { get; set; } = 0.01;
        public double Lambda { get; set; } = 0.1;
        public double Epsilon { get; set; } = 0.01;

        public int CurrentState { get; private set; }

        private readonly int numStates;
        private readonly int numActions;

        public QAgent(double[,] qtable, QAlgorithm algorithm = QAlgorithm.Q, TraceType traceType = TraceType.None, Random random = null)
        {
            QTable = qtable;
            Algorithm = algorithm;
            TraceType = traceType;
            Random = random ?? new Random();
            Traces = new double[qtable.GetLength(0), qtable.GetLength(1)];
            CurrentState = 0;

            numStates = qtable.GetLength(0);
            numActions = qtable.GetLength(1);
        }

        public void Reset(int state)
        {
            CurrentState = state;
            if (TraceType != TraceType.None)
                ResetTraces();
        }

        public void ResetTraces()
        {
            for (var s = 0; s < numStates; s++)
            for (var a = 0; a < numActions; a++)
                Traces[s, a] = 0.0;
        }

        public int GetGreedyAction(int state)
        {
            var bestAction = Random.Next(numActions);
            var bestQ = QTable[state, bestAction];

            for (int a = 0; a < numActions; a++)
            {
                if (QTable[state, a] > bestQ)
                {
                    bestQ = QTable[state, a];
                    bestAction = a;
                }
            }

            return bestAction;
        }

        public int GetPolicyAction(int state)
        {
            if (Random.NextDouble() < Epsilon)
                return Random.Next(numActions);
            return GetGreedyAction(state);
        }

        public int Step(int state, int action, double reward, int nextState)
        {
            if (state != CurrentState && TraceType != TraceType.None)
                ResetTraces();

            var greedyAction = GetGreedyAction(nextState);
            var policyAction = GetPolicyAction(nextState);
            var updateAction = Algorithm == QAlgorithm.Q ? greedyAction : policyAction;
            var delta = reward + Gamma * QTable[nextState, updateAction] - QTable[state, action];
            var decay = updateAction == policyAction ? Lambda * Gamma : 0;

            Act(state, action);
            Update(state, action, delta, decay);

            CurrentState = nextState;
            return policyAction;
        }

        private void Act(int state, int action)
        {
            if (TraceType == TraceType.Replacing)
                Traces[state, action] = 1;
            else if (TraceType == TraceType.Accumulating)
                Traces[state, action] += 1;
        }

        private void Update(int state, int action, double delta, double decay)
        {
            if (TraceType != TraceType.None)
            {
                if (numStates > numActions)
                {
                    Parallel.For(0, numStates, s =>
                    {
                        for (var a = 0; a < numActions; a++)
                        {
                            QTable[s, a] += Alpha * delta * Traces[s, a];
                            Traces[s, a] = decay * Traces[s, a];
                        }
                    });
                }
                else
                {
                    Parallel.For(0, numActions, a =>
                    {
                        for (var s = 0; s < numStates; a++)
                        {
                            QTable[s, a] += Alpha * delta * Traces[s, a];
                            Traces[s, a] = decay * Traces[s, a];
                        }
                    });
                }
            }
            else
            {
                QTable[state, action] += Alpha * delta;
            }
        }

        public void Reward(int state, int action, double reward)
        {
            Update(state, action, reward, 1.0);
        }
    }
}