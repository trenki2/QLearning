using System;
using System.Diagnostics;

namespace QLearning
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            Test();
        }

        private static void Test()
        {
            var sw = Stopwatch.StartNew();
            var random = new Random();

            var table = new double[720, 3];
            var agent = new QAgent(table, QAlgorithm.Sarsa, traces: false);

            agent.Alpha = 0.5;
            agent.Gamma = 0.99;
            agent.Lambda = 0.5;

            var reward = new double[720, 3];
            for (int state = 0; state < 720; state++)
            {
                var action = random.Next(2);
                reward[state, action] = 1.0;
            }

            for (var epoch = 0; epoch < 100; epoch++)
            {
                agent.Reset(0);
                agent.Epsilon = 1 - epoch / 100.0;

                var action = agent.GetPolicyAction(0);
                var cumulativeReward = 0.0;

                for (var state = 0; state < 720 - 1; state++)
                {
                    cumulativeReward += reward[state, action];
                    action = agent.Step(state, action, reward[state, action], state + 1);
                }

                Console.WriteLine($"{cumulativeReward / table.GetLength(0):0.000}");
            }

            Console.WriteLine(sw.ElapsedMilliseconds);
        }
    }
}