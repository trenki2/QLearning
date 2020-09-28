using System;
using System.Diagnostics;
using System.Linq;
using System.Xml.Schema;

namespace QLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            var sw = Stopwatch.StartNew();

            var random = new Random();
           
            var table = new QTable(16, 16);
            var learner = new QLearner(table, QAlgorithm.Sarsa);
            learner.Random = random;

            var gammaTable = new QTable(1, 4);
            var gammaLearner = new QLearner(gammaTable, QAlgorithm.Sarsa);
            gammaLearner.Random = random;
            gammaLearner.Epsilon = 0.05;
            gammaLearner.Temperature = 0.5;

            var alphaTable = new QTable(1, 4);
            var alphaLearner = new QLearner(alphaTable, QAlgorithm.Sarsa);
            alphaLearner.Random = random;
            alphaLearner.Epsilon = 0.05;
            alphaLearner.Temperature = 0.5;

            var epsilonTable = new QTable(1, 4);
            var epsilonLearner = new QLearner(epsilonTable, QAlgorithm.Sarsa);
            epsilonLearner.Random = random;
            epsilonLearner.Epsilon = 0.05;
            epsilonLearner.Temperature = 0.5;

            var temperatureTable = new QTable(1, 4);
            var temperatureLearner = new QLearner(temperatureTable, QAlgorithm.Sarsa);
            temperatureLearner.Random = random;
            temperatureLearner.Epsilon = 0.05;
            temperatureLearner.Temperature = 0.5;

            var epochs = 100;
            var steps = 1000;
            var epsilon = 0.2;
            var totalReward = 0.0;

            var gammaAction = 0;
            var alphaAction = 0;
            var epsilonAction = 0;
            var temperatureAction = 0;

            for (var epoch = 0; epoch < epochs; epoch++)
            {
                var action = 0;
                var reward = 0.0;
                var epochReward = 0.0;

                for (var i = 0; i < steps; i++)
                {
                    var state = learner.CurrentState;
                    var nextState = random.NextDouble() < epsilon ? random.Next(table.GetStateCount()) : state;

                    reward = action == nextState ? 1 : 0;
                    action = learner.Learn(state, action, reward, nextState);

                    totalReward += reward;
                    epochReward += reward;
                }

                gammaAction = alphaLearner.Learn(gammaLearner.CurrentState, gammaAction, epochReward / steps, gammaLearner.CurrentState);

                switch (gammaAction)
                {
                    case 0: learner.Gamma = 0.3; break;
                    case 1: learner.Gamma = 0.7; break;
                    case 2: learner.Gamma = 0.9; break;
                    case 3: learner.Gamma = 0.99; break;
                }

                alphaAction = alphaLearner.Learn(alphaLearner.CurrentState, alphaAction, epochReward / steps, alphaLearner.CurrentState);
                
                switch (alphaAction)
                {
                    case 0: learner.Alpha = 0.01; break;
                    case 1: learner.Alpha = 0.05; break;
                    case 2: learner.Alpha = 0.10; break;
                    case 3: learner.Alpha = 0.15; break;
                }

                epsilonAction = alphaLearner.Learn(epsilonLearner.CurrentState, epsilonAction, epochReward / steps, epsilonLearner.CurrentState);

                switch (epsilonAction)
                {
                    case 0: learner.Epsilon = 0.01; break;
                    case 1: learner.Epsilon = 0.03; break;
                    case 2: learner.Epsilon = 0.05; break;
                    case 3: learner.Epsilon = 0.08; break;
                }

                temperatureAction = temperatureLearner.Learn(temperatureLearner.CurrentState, temperatureAction, epochReward / steps, temperatureLearner.CurrentState);

                switch (temperatureAction)
                {
                    case 0: learner.Temperature = 0.1; break;
                    case 1: learner.Temperature = 0.05; break;
                    case 2: learner.Temperature = 0.01; break;
                    case 3: learner.Temperature = 0.005; break;
                }

                Console.WriteLine($"{epoch}; {gammaAction}; {alphaAction}; {epsilonAction}; {temperatureAction}; {epochReward / steps}");

                //for (var s = 0; s < table.GetStateCount(); s++)
                //{
                //    for (var a = 0; a < table.GetActionCount(s); a++)
                //        Console.Write($"{table[s, a]:0.00}; ");
                //    Console.WriteLine();
                //}
            }

            Console.WriteLine($"{totalReward / (steps * epochs)}");
            Console.WriteLine(sw.ElapsedMilliseconds);
        }
    }
}
