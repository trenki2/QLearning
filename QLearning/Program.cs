using System;
using System.Diagnostics;
using System.Threading.Tasks.Sources;

namespace QLearning
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            //Test1();
            Test2();
        }

        private static void Test1()
        {
            var sw = Stopwatch.StartNew();
            var random = new Random();

            var table = new double[720, 3];
            var agent = new QAgent(table, QAlgorithm.Sarsa);

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

        private static void Test2()
        {
            var table = new double[2000, 10000];
            var agent = new QAgent(table, QAlgorithm.Sarsa, TraceType.Replacing) { Gamma = 0.99, Lambda = 0.99 };
            var random = new Random();

            var sw = Stopwatch.StartNew();
            var state = 0;
            var action = 0;
            for (int i = 0; i < 100000; i++)
            {
                var nextState = random.Next(table.GetLength(0));
                action = agent.Step(state, action, 0, nextState);
                state = nextState;
            }

            Console.WriteLine(sw.ElapsedMilliseconds);
        }
    }
}