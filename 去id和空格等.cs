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
            getPredict(@"\Users\apple\Desktop\NLP\textclassfier\text\");
            //D:\python\textclassfier\text\
        }
       
        public static void getPredict(string nowpath)
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
                        for (int k = 0; k < DataArr.Length; k++)
                        {
                            
                            //Console.WriteLine(DataArr[k].IndexOf(" "));
                            //Console.WriteLine(DataArr[k].IndexOf("\t"));
                            int index = -1;
                            if (DataArr[k].IndexOf(" ") < DataArr[k].IndexOf("\t") && DataArr[k].IndexOf(" ") > 0)
                            {
                                index = DataArr[k].IndexOf(" ");

                            }
                            else
                            {
                                index = DataArr[k].IndexOf("\t");
                            }
                            if(index>21 || index < 15)
                            {
                                Console.WriteLine(DataArr[k].IndexOf(" "));
                                Console.WriteLine(DataArr[k].IndexOf("\t"));
                                Console.WriteLine(DataArr[k]);
                            }
                            DataArr[k] = DataArr[k].Substring(index).Trim().Replace("\t", "").Replace(" ", "");
                            
                        }
                        string filePath = @"\Users\apple\Desktop\NLP\textclassfier\text\newdata\" + NextFile.Name;
                    if (File.Exists(filePath))
                            File.Delete(filePath);
                        //如果文件不存在，则创建；存在则覆盖
                        //该方法写入字符数组换行显示                   
                        System.IO.File.WriteAllLines(filePath, DataArr, Encoding.UTF8);

                    }

                  
                }
             

            }
            catch
            {

            }
           
        }
    }
}