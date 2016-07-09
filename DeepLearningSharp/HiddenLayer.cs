using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningSharp
{
    public class HiddenLayer
    {
        public int N;
        public int n_in;
        public int n_out;
        public double[][] W;
        public double[] b;
        public Random rng;
        public Func<double, double> activation;
        public Func<double, double> dactivation;

        public HiddenLayer(int N, int n_in, int n_out, double[][] W, double[] b, Random rng, String activation)
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

            if (activation == "sigmoid" || activation == null)
            {
                //this.activation = (double x) -> utils.sigmoid(x);
                //this.dactivation = (double x) -> utils.dsigmoid(x);
                this.activation = utils.sigmoid;
                this.dactivation = utils.dsigmoid;

            }
            else if (activation == "tanh")
            {
                //this.activation = (double x) -> utils.tanh(x);
                //this.dactivation = (double x) -> utils.dtanh(x);
                this.activation = utils.tanh;
                this.dactivation = utils.dtanh;
            }
            else if (activation == "ReLU")
            {
                //this.activation = (double x) -> utils.ReLU(x);
                //this.dactivation = (double x) -> utils.dReLU(x);
                this.activation = utils.ReLU;
                this.dactivation = utils.dReLU;
            }
            else
            {
                throw new /*IllegalArgument*/Exception("activation function not supported");
            }

        }

        public double output(double[] input, double[] w, double b)
        {
            double linear_output = 0.0;
            for (int j = 0; j < n_in; j++)
            {
                linear_output += w[j] * input[j];
            }
            linear_output += b;

            return activation(linear_output);
        }


        public void forward(double[] input, double[] output)
        {
            for (int i = 0; i < n_out; i++)
            {
                output[i] = this.output(input, W[i], b[i]);
            }
        }

        public void backward(double[] input, double[] dy, double[] prev_layer_input, double[] prev_layer_dy, double[][] prev_layer_W, double lr)
        {
            if (dy == null) dy = new double[n_out];

            int prev_n_in = n_out;
            int prev_n_out = prev_layer_dy.Length;

            for (int i = 0; i < prev_n_in; i++)
            {
                dy[i] = 0;
                for (int j = 0; j < prev_n_out; j++)
                {
                    dy[i] += prev_layer_dy[j] * prev_layer_W[j][i];
                }

                dy[i] *= dactivation(prev_layer_input[i]);
            }

            for (int i = 0; i < n_out; i++)
            {
                for (int j = 0; j < n_in; j++)
                {
                    W[i][j] += lr * dy[i] * input[j] / N;
                }
                b[i] += lr * dy[i] / N;
            }
        }

        public int[] dropout(int size, double p, Random rng)
        {
            int[] mask = new int[size];

            for (int i = 0; i < size; i++)
            {
                mask[i] = utils.binomial(1, p, rng);
            }

            return mask;
        }
    }

}
