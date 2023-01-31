using System;
using System.Collections.Generic;
using System.Linq;

namespace QLearning
{
    public class QAgent
    {
        public double[,] QTable { get; set; }
        public Dictionary<(int State, int Action), double> Traces { get; set; } = new();
        public QAlgorithm Algorithm { get; set; }
        public TraceType TraceType { get; set; }
        public double TraceThreshold { get; set; } = 1e-4;
        public Random Random { get; set; }

        public double Gamma { get; set; } = 0.99;
        public double Lambda { get; set; } = 0.99;
        public double Alpha { get; set; } = 0.01;
        public double Epsilon { get; set; } = 0.01;

        public int CurrentState { get; private set; }

        public int NumStates { get; private set; }
        public int NumActions { get; private set; }

        private int updateCount;
        private (int Action, double QValue)[] bestActions;

        public QAgent(double[,] qtable, QAlgorithm algorithm = QAlgorithm.Q, TraceType traceType = TraceType.Replacing, Random random = null)
        {
            QTable = qtable;
            Algorithm = algorithm;
            TraceType = traceType;
            Random = random ?? new Random();
            CurrentState = 0;
            NumStates = qtable.GetLength(0);
            NumActions = qtable.GetLength(1);
            bestActions = Enumerable.Range(0, NumStates).Select(x => (Random.Next(NumActions), -double.MaxValue)).ToArray();
        }

        public void Reset(int state)
        {
            CurrentState = state;
            Traces.Clear();
        }

        public int GetGreedyAction(int state)
        {
            if (bestActions[state].Action != -1)
                return bestActions[state].Action;

            var bestAction = Random.Next(NumActions);
            var bestQ = QTable[state, bestAction];

            for (int a = 0; a < NumActions; a++)
            {
                if (QTable[state, a] > bestQ)
                {
                    bestQ = QTable[state, a];
                    bestAction = a;
                }
            }

            bestActions[state] = (bestAction, bestQ);
            return bestAction;
        }

        public int GetPolicyAction(int state, int? greedyAction = null)
        {
            if (Random.NextDouble() < Epsilon)
                return Random.Next(NumActions);
            return greedyAction ?? GetGreedyAction(state);
        }

        public int Step(int state, int action, double reward, int nextState)
        {
            if (state != CurrentState && TraceType != TraceType.None)
                Traces.Clear();

            var greedyAction = GetGreedyAction(nextState);
            var policyAction = GetPolicyAction(nextState, greedyAction);
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
            {
                Traces[(state, action)] = 1;
            }
            else if (TraceType == TraceType.Accumulating)
            {
                Traces.TryGetValue((state, action), out var val);
                Traces[(state, action)] = val + 1;
            }
        }

        private void Update(int state, int action, double delta, double decay)
        {
            if (TraceType == TraceType.None)
            {
                QTable[state, action] += Alpha * delta;
                UpdateBestAction(state, action);
                return;
            }

            foreach (var key in Traces.Keys)
            {
                QTable[key.State, key.Action] += Alpha * delta * Traces[key];
                UpdateBestAction(state, action);
                Traces[key] *= decay;
            }

            if (++updateCount % 100 == 0)
            {
                foreach (var key in Traces.Where(x => x.Value < TraceThreshold).Select(x => x.Key).ToArray())
                    Traces.Remove(key);
            }
        }

        private void UpdateBestAction(int state, int action)
        {
            if (QTable[state, action] > bestActions[state].QValue)
                bestActions[state] = (action, QTable[state, action]);
            else if (bestActions[state].Action == action && QTable[state, action] < bestActions[state].QValue)
                bestActions[state] = (-1, -double.MaxValue);
        }

        public void Reward(int state, int action, double reward)
        {
            Update(state, action, reward, 1.0);
        }
    }
}