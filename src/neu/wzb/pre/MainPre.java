package neu.wzb.pre;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class MainPre {

	public static void main(String[] args) {
		NeutralClas neutral=new NeutralClas();
		Instances instancesTrain =neutral.getInstances("D:/测试/Elec/ElectricNormal/normal.arff");
		instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);
		Instances instancesTest=neutral.getInstances("D:/测试/Elec/ElectricNormal/normal1.arff");
		instancesTest.setClassIndex(instancesTrain.numAttributes()-1);
		double sum = instancesTest.numInstances();
		MultilayerPerceptron m_classifier=	neutral.MuPreceptron(instancesTrain);
		for (int i = 0; i < sum; i++) {
			try {
				double predited = m_classifier.classifyInstance(instancesTest
						.instance(i));
				System.out.println("预测某条记录的分类Id:" + predited + ",分类值："
						+ instancesTest.classAttribute().value((int) predited));
				System.out.println(m_classifier.classifyInstance(instancesTest
						.instance(i)));
				System.out.println("测试文件的分类值:"
						+ instancesTest.instance(i).classValue() + ",记录:"
						+ instancesTest.instance(i));
				System.out.println("************************************");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}

	}

