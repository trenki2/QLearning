using System;
using System.Collections.Generic;

namespace QLearning
{
    public class QTable
    {
        private readonly List<double[]> data;

        public QTable() : this(0, 0)
        {

        }

        public QTable(QTable other) : this(0, 0)
        {
            this.data = new List<double[]>(other.GetStateCount());
            for (var s = 0; s < other.GetStateCount(); s++)
                AddState(other.GetActionCount(s));
        }

        public QTable(int states, int actions)
        {
            this.data = new List<double[]>(states);
            for (var s = 0; s < states; s++)
                AddState(actions);
        }

        public int GetStateCount() => data.Count;
        public int GetActionCount(int state) => data[state].Length;

        public double this[int state, int action]
        {
            get => data[state][action];
            set => data[state][action] = value;
        }

        public void AddState(int actions)
        {
            data.Add(new double[actions]);
        }
    }
}
