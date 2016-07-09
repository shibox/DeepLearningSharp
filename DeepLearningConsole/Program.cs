using DeepLearningSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            SdA.main(args);

            RBM.main(args);

            MLP.main(args);

            LogisticRegressionDiscrete.main(args);

            LogisticRegression.main(args);

            Dropout.main(args);

            DBN.main(args);

            dA.main(args);

            Console.WriteLine("finish");
            Console.ReadLine();
        }
    }
}
