using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp2
{
    class Program
    {
        static void Main(string[] args)
        {
            getPredict(@"D:\python\textclassfier\text\data_fenci\",0.7);
        }

        public static void getPredict(string nowpath,double percent)
        {
            try
            {
                int n = 0;
                DirectoryInfo nowDataFolder = new DirectoryInfo(nowpath);
                if (nowDataFolder.GetFiles().Length != 0)
                {
                    string[] DataArr = null;
                    foreach (FileInfo NextFile in nowDataFolder.GetFiles())
                    {
                        Console.WriteLine(NextFile.FullName);
                        DataArr = File.ReadAllLines(NextFile.FullName);
                        int trainNum = (int)Math.Round(DataArr.Length * percent);
                        int testNum = DataArr.Length - trainNum;
                        string[] trainData = new string[trainNum];
                        string[] testData = new string[testNum];
                        for(int i = 0; i < trainNum; i++)
                        {
                            trainData[i] = DataArr[i];
                        }
                        for(int j = trainNum; j < DataArr.Length; j++)
                        {
                            testData[j - trainNum] = DataArr[j];
                        }
                        Console.WriteLine(trainData[111]);
                        Console.WriteLine(trainData[112]);
                        Console.WriteLine(trainData[113]);
                        Console.WriteLine(trainData[114]);
                        Console.ReadKey();
                        string trainFilePath = @"D:\python\textclassfier\text\data_fenci\finalData\" + "trainData.txt";
                        string testFilePath = @"D:\python\textclassfier\text\data_fenci\finalData\" + "testData.txt";

                        //System.IO.File.WriteAllLines(trainFilePath, DataArr, Encoding.UTF8);
                        //System.IO.File.WriteAllLines(testFilePath, DataArr, Encoding.UTF8);
                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(trainFilePath, true))
                        {
                            foreach (string line in trainData)
                            {
                                    file.WriteLine(line);// 直接追加文件末尾，换行                                
                            }
                        }
                        using (System.IO.StreamWriter file = new System.IO.StreamWriter(testFilePath, true))
                        {
                            foreach (string line in testData)
                            {
                                file.WriteLine(line);// 直接追加文件末尾，换行                                
                            }
                        }

                    }
                }
            }
            catch
            {

            }

        }
    }
}