package neu.wzb.pre;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;

public class MainPre {

	public static void main(String[] args) {
		NeutralClas neutral=new NeutralClas();
		Instances instancesTrain =neutral.getInstances("D:/����/Elec/ElectricNormal/normal.arff");
		instancesTrain.setClassIndex(instancesTrain.numAttributes()-1);
		Instances instancesTest=neutral.getInstances("D:/����/Elec/ElectricNormal/normal1.arff");
		instancesTest.setClassIndex(instancesTrain.numAttributes()-1);
		double sum = instancesTest.numInstances();
		MultilayerPerceptron m_classifier=	neutral.MuPreceptron(instancesTrain);
		for (int i = 0; i < sum; i++) {
			try {
				double predited = m_classifier.classifyInstance(instancesTest
						.instance(i));
				System.out.println("Ԥ��ĳ����¼�ķ���Id:" + predited + ",����ֵ��"
						+ instancesTest.classAttribute().value((int) predited));
				System.out.println(m_classifier.classifyInstance(instancesTest
						.instance(i)));
				System.out.println("�����ļ��ķ���ֵ:"
						+ instancesTest.instance(i).classValue() + ",��¼:"
						+ instancesTest.instance(i));
				System.out.println("************************************");
			} catch (Exception e) {
				e.printStackTrace();
			}
		}

	}

	}

