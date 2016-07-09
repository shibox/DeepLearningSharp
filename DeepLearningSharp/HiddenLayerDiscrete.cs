using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningSharp
{
    public class HiddenLayerDiscrete : HiddenLayer
    {
        public int N;
        public int n_in;
        public int n_out;
        public double[][] W;
        public double[] b;
        public Random rng;


        public HiddenLayerDiscrete(int N, int n_in, int n_out, double[][] W, double[] b, Random rng)
                : base(N, n_in, n_out, W, b, rng, null)
        {


            this.N = N;
            this.n_in = n_in;
            this.n_out = n_out;

            if (rng == null) this.rng = new Random(1234);
            else this.rng = rng;

            if (W == null)
            {
                this.W = utils.CreateCrossArray<double>(n_out, n_in);// new double[n_out,n_in];
                double a = 1.0 / this.n_in;

                for (int i = 0; i < n_out; i++)
                {
                    for (int j = 0; j < n_in; j++)
                    {
                        this.W[i][j] = utils.uniform(-a, a, rng);
                    }
                }
            }
            else
            {
                this.W = W;
            }

            if (b == null) this.b = new double[n_out];
            else this.b = b;
        }

        public double output(int[] input, double[] w, double b)
        {
            double linear_output = 0.0;
            for (int j = 0; j < n_in; j++)
            {
                linear_output += w[j] * input[j];
            }
            linear_output += b;
            return utils.sigmoid(linear_output);
        }

        public void sample_h_given_v(int[] input, int[] sample)
        {
            for (int i = 0; i < n_out; i++)
            {
                sample[i] = utils.binomial(1, output(input, W[i], b[i]), rng);
            }
        }
    }
}
