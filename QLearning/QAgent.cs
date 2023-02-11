using System;
using System.Collections.Generic;
using System.Linq;

namespace QLearning
{
    public class QAgent
    {
        public double[,] QTable { get; set; }
        public double[,] Traces { get; set; }
        public Dictionary<(int State, int Action), double> TracesDict { get; set; } = new();
        public QAlgorithm Algorithm { get; set; }
        public TraceType TraceType { get; set; }
        public TraceMode TraceMode { get; set; }
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

        public QAgent(double[,] qtable, QAlgorithm algorithm = QAlgorithm.Q, TraceType traceType = TraceType.Replacing, TraceMode traceMode = TraceMode.Matrix, Random random = null)
        {
            QTable = qtable;
            Traces = new double[qtable.GetLength(0), qtable.GetLength(1)];
            Algorithm = algorithm;
            TraceType = traceType;
            TraceMode = traceMode;
            Random = random ?? new Random();
            CurrentState = 0;
            NumStates = qtable.GetLength(0);
            NumActions = qtable.GetLength(1);
            bestActions = Enumerable.Range(0, NumStates).Select(x => (Random.Next(NumActions), -double.MaxValue)).ToArray();
        }

        public void Reset(int state)
        {
            CurrentState = state;
            ResetTraces();
        }

        public void ResetTraces()
        {
            if (TraceMode == TraceMode.Matrix)
            {
                for (int s = 0; s < NumStates; s++)
                    for (int a = 0; a < NumActions; a++)
                        Traces[s, a] = 0;
            }
            else
            {
                TracesDict.Clear();
            }
        }

        public int GetGreedyAction(int state)
        {
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
                ResetTraces();

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

        public void Act(int state, int action)
        {
            if (TraceType == TraceType.Replacing)
                SetTrace(state, action, 1);
            else if (TraceType == TraceType.Accumulating)
                AddTrace(state, action, 1);
        }

        public void Update(int state, int action, double delta, double decay)
        {
            if (TraceType == TraceType.None)
            {
                QTable[state, action] += Alpha * delta;
                return;
            }

            if (TraceMode == TraceMode.Matrix)
            {
                for (var s = 0; s < NumStates; s++)
                {
                    for (var a = 0; a < NumActions; a++)
                    {
                        QTable[s, a] += Alpha * delta * Traces[s, a];
                        Traces[s, a] *= decay;
                    }
                }
            }
            else
            {
                foreach (var key in TracesDict.Keys)
                {
                    QTable[key.State, key.Action] += Alpha * delta * GetTrace(state, action);
                    MulTrace(state, action, decay);
                }

                if (++updateCount % 100 == 0)
                {
                    foreach (var key in TracesDict.Where(x => x.Value < TraceThreshold).Select(x => x.Key).ToArray())
                        TracesDict.Remove(key);
                }
            }
        }

        public void Reward(int state, int action, double reward)
        {
            Update(state, action, reward, 1.0);
        }

        private void SetTrace(int state, int action, double value)
        {
            if (TraceMode == TraceMode.Matrix)
                Traces[state, action] = value;
            else
                TracesDict[(state, action)] = value;
        }

        private void AddTrace(int state, int action, double value)
        {
            if (TraceMode == TraceMode.Matrix)
            {
                Traces[state, action] += value;
            }
            else
            {
                TracesDict.TryGetValue((state, action), out var val);
                TracesDict[(state, action)] = val + value;
            }
        }

        private void MulTrace(int state, int action, double value)
        {
            if (TraceMode == TraceMode.Matrix)
            {
                Traces[state, action] *= value;
            }
            else
            {
                TracesDict.TryGetValue((state, action), out var val);
                TracesDict[(state, action)] = val * value;
            }
        }

        private double GetTrace(int state, int action)
        {
            if (TraceMode == TraceMode.Matrix)
            {
                return Traces[state, action];
            }
            else
            {
                TracesDict.TryGetValue((state, action), out var value);
                return value;
            }
        }
    }
}