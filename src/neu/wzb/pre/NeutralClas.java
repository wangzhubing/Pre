package neu.wzb.pre;

import java.io.File;
import java.io.IOException;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instances;
import weka.core.converters.ArffLoader;


public class NeutralClas {
	public Instances getInstances(String st){
		File inputFile = new File(st);
		ArffLoader atf = new ArffLoader();
		try {
			atf.setFile(inputFile);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		Instances instances= null;
		try {
			instances = atf.getDataSet();
		} catch (IOException e) {

			e.printStackTrace();
		}
		return instances;
	}
	public MultilayerPerceptron MuPreceptron(Instances instancesTrain){
		MultilayerPerceptron m_classifier = new MultilayerPerceptron();
		m_classifier.setTrainingTime(200);
		m_classifier.setLearningRate(0.01);
		m_classifier.setValidationThreshold(20);
		m_classifier.setNormalizeAttributes(true);
		m_classifier.setNormalizeNumericClass(true);
		try {
			m_classifier.buildClassifier(instancesTrain);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return m_classifier;
	}
}
