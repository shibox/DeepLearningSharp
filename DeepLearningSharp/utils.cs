using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningSharp
{
    public class utils
    {
        public static T[][] CreateCrossArray<T>(int x, int y)
        {
            T[][] result = new T[x][];
            for (int i = 0; i < x; i++)
                result[i] = new T[y];
            return result;
        }

        public static double uniform(double min, double max, Random rng)
        {
            return rng.NextDouble() * (max - min) + min;
        }

        public static int binomial(int n, double p, Random rng)
        {
            if (p < 0 || p > 1) return 0;

            int c = 0;
            double r;

            for (int i = 0; i < n; i++)
            {
                r = rng.NextDouble();
                if (r < p) c++;
            }

            return c;
        }

        public static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -x));
        }

        public static double dsigmoid(double x)
        {
            return x * (1.0 - x);
        }

        public static double tanh(double x)
        {
            return Math.Tanh(x);
        }

        public static double dtanh(double x)
        {
            return 1.0 - x * x;
        }

        public static double ReLU(double x)
        {
            if (x > 0)
            {
                return x;
            }
            else
            {
                return 0.0;
            }
        }

        public static double dReLU(double x)
        {
            if (x > 0)
            {
                return 1.0;
            }
            else
            {
                return 0.0;
            }
        }
    }
}
