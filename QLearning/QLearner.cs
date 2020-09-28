using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace QLearning
{
    public class QLearner
    {
        private QTable e;

        public QTable Q { get; }
        public QAlgorithm Algorithm { get; }
        
        public Random Random { get; set; } = new Random();

        public double Gamma { get; set; } = 0.9;
        public double Alpha { get; set; } = 0.01;
        public double Lambda { get; set; } = 0.1;
        public double Epsilon { get; set; } = 0.01;
        public double Temperature { get; set; } = 0.1;

        public int CurrentState { get; private set; }

        public QLearner(QTable Q, QAlgorithm algorithm)
        {
            this.Q = Q;
            this.Algorithm = algorithm;

            e = new QTable(Q);

            ResetState(0);
        }

        public void ResetState(int newState)
        {
            CurrentState = newState;
            ResetEligibilityTraces();
        }

        public void ResetEligibilityTraces()
        {
            for (var s = 0; s < e.GetStateCount(); s++)
            for (var a = 0; a < e.GetActionCount(s); a++)
                e[s, a] = 0.0;
        }

        public int Learn(int state, int action, double reward, int nextState)
        {
            if (state != CurrentState)
                ResetEligibilityTraces();

            var greedyAction = GetGreedyAction(nextState);
            var policyAction = GetPolicyAction(nextState);
            var updateAction = Algorithm == QAlgorithm.Q ? greedyAction : policyAction;

            var delta = reward + Gamma * Q[nextState, updateAction] - Q[state, action];
            e[state, action] = 1;

            for (int s = 0; s < Q.GetStateCount(); s++)
            {
                var maxQ = Q[s, 0];
                for (int a = 1; a < Q.GetActionCount(s); a++)
                    maxQ = Math.Max(maxQ, Q[s, a]);

                var sum = 0.0;
                for (int a = 0; a < Q.GetActionCount(s); a++)
                    sum += Math.Exp((Q[s, a] - maxQ) / Temperature);
                if (sum < double.Epsilon)
                    sum = double.Epsilon;

                for (int a = 0; a < Q.GetActionCount(s); a++)
                {
                    var pi = GetPolicy(s, a, maxQ, sum);
                    if (pi < double.Epsilon)
                        pi = double.Epsilon;

                    Q[s, a] = Q[s, a] + Math.Pow(1 - Alpha, 1.0 / pi) * delta * e[s, a];
                    e[s, a] = policyAction == updateAction ? Gamma * Lambda * e[s, a] : 0.0;
                }
            }

            CurrentState = nextState;
            return policyAction;
        }

        private double GetPolicy(int state, int action, double maxQ, double sum)
        {
            return Math.Exp((Q[state, action] - maxQ) / Temperature) / sum;
        }

        public double GetPolicy(int state, int action)
        {
            var maxQ = Q[state, 0];
            for (int a = 1; a < Q.GetActionCount(state); a++)
                maxQ = Math.Max(maxQ, Q[state, a]);

            var sum = 0.0;
            for (int a = 0; a < Q.GetActionCount(state); a++)
                sum += Math.Exp((Q[state, a] - maxQ) / Temperature);
            sum = Math.Max(sum, double.Epsilon);

            return GetPolicy(state, action, maxQ, sum);
        }

        public unsafe int GetPolicyAction(int state)
        {
            if (Epsilon > 0.0 && Random.NextDouble() < Epsilon)
                return Random.Next(Q.GetActionCount(state));

            var bound = stackalloc double[Q.GetActionCount(state)];
            var val = 0.0;

            var maxQ = Q[state, 0];
            for (int a = 1; a < Q.GetActionCount(state); a++)
                maxQ = Math.Max(maxQ, Q[state, a]);

            var sum = 0.0;
            for (int a = 0; a < Q.GetActionCount(state); a++)
                sum += Math.Exp((Q[state, a] - maxQ) / Temperature);
            sum = Math.Max(sum, double.Epsilon);

            for (var action = 0; action < Q.GetActionCount(state); action++)
            {
                val += GetPolicy(state, action, maxQ, sum);
                bound[action] = val;
            }

            var r = Random.NextDouble() * val;
            for (var action = 0; action < Q.GetActionCount(state); action++)
            {
                if (r <= bound[action])
                    return action;
            }

            return Q.GetActionCount(state) - 1;
        }

        public int GetGreedyAction(int state)
        {
            var bestAction = Random.Next(Q.GetActionCount(state));
            var bestQ = Q[state, bestAction];

            for (int a = 0; a < Q.GetActionCount(state); a++)
            {
                if (Q[state, a] > bestQ)
                {
                    bestQ = Q[state, a];
                    bestAction = a;
                }
            }

            return bestAction;
        }

        public int GetEpsilonGreedyAction(int state, int greedyAction = -1)
        {
            var action = greedyAction;
            if (action == -1)
                action = GetGreedyAction(state);

            if (Random.NextDouble() < Epsilon)
                action = Random.Next(Q.GetActionCount(state));

            return action;
        }
    }
}
