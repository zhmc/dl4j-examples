package org.deeplearning4j.examples.nlp.paragraphvectors;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.examples.miscUtil.GBKFileLabelAwareIterator;
import org.deeplearning4j.examples.miscUtil.HanlpTokenizerFactory;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.LabelSeeker;
import org.deeplearning4j.examples.nlp.paragraphvectors.tools.MeansBuilder;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.util.List;

/**
 * 本文主要是用ParagraphVectors方法做文档分类，训练数据有一些带类别的文档，
 * 预测没有类别的文档属于哪个类别。这里简单说下ParagraphVectors模型，每篇
 * 文档映射在一个唯一的向量上，由矩阵中的一列表示，每个word则类似的被映射到
 * 向量上，这个向量由另一个矩阵的列表示。使用连接方式获得新word的预测，可以
 * 说ParagraphVectors是在word2vec基础上加了一组paragraph输入列向量一起训
 * 练构成的模型
 */
public class PVHotel {

	ParagraphVectors paragraphVectors; //声明ParagraphVectors类
	
	//声明要实现的迭代器接口，用来识别句子或文档及标签，这里假定所有的文档已变成字符串或词表的形式
	LabelAwareIterator iterator;       
	
	TokenizerFactory tokenizerFactory; //声明字符串分割器    

	private static final Logger log = LoggerFactory.getLogger(PVHotel.class);

	public static void main(String[] args) throws Exception {
		
		
		PVHotel app = new PVHotel();//又是这种写法，构建实现类    
		
//		app.tokenizerFactory.
		
		app.makeParagraphVectors();     //调用构建模型方法      
		app.checkUnlabeledData();       //检查标签数据 
		/*
		 * Your output should be like this:
		 * 
		 * Document 'health' falls into the following categories: health:
		 * 0.29721372296220205 science: 0.011684473733853906 finance:
		 * -0.14755302887323793
		 * 
		 * Document 'finance' falls into the following categories: health:
		 * -0.17290237675941766 science: -0.09579267574606627 finance:
		 * 0.4460859189453788
		 * 
		 * so,now we know categories for yet unseen documents
		 */
	}

	
	void makeParagraphVectors() throws Exception {
		ClassPathResource resource = new ClassPathResource("ChnSentiCorp_htl_unba_10000/labeled");//弄一个带标签的文档路径    

		// build a iterator for our dataset
		//实现LabelAwareIterator接口，添加数据源，构成迭代器           
		iterator = new GBKFileLabelAwareIterator.Builder().addSourceFolder(resource.getFile()).build();

		tokenizerFactory = new HanlpTokenizerFactory();//构建逗号分割器    
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

		// ParagraphVectors training configuration
		
		//ParagraphVectors继承Word2Vec，Word2Vec继承SequenceVectors，
		//配置ParagraphVectors的学习率，最小学习率，批大小，步数，迭代器，同时构建词和文档，词分割器
		paragraphVectors = new ParagraphVectors.Builder()
				.learningRate(0.025)
				.minLearningRate(0.001)
				.batchSize(8000)
				.epochs(20)
				.iterate(iterator)
				.trainWordVectors(true)
				.tokenizerFactory(tokenizerFactory).build();
		
		
		//模型定型
		// Start model training
		paragraphVectors.fit();
		
	}

	void checkUnlabeledData() throws FileNotFoundException {
		
		//这里假定模型已经构建好，现在预测无标签的文档属于哪个类，我们装载无标签文档并对其进行检测      
		/*
		 * At this point we assume that we have model built and we can check
		 * which categories our unlabeled document falls into. So we'll start
		 * loading our unlabeled documents and checking them
		 */
		
		//构建无标签文档读取器     
		ClassPathResource unClassifiedResource = new ClassPathResource("ChnSentiCorp_htl_unba_10000/unlabeled");
		GBKFileLabelAwareIterator unClassifiedIterator = new GBKFileLabelAwareIterator.Builder()
				.addSourceFolder(unClassifiedResource.getFile()).build();

		//预测未标记文档，很多情况一个文档可能对应多个类别，只不过每个类别值有高有低 
		/*
		 * Now we'll iterate over unlabeled data, and check which label it could
		 * be assigned to Please note: for many domains it's normal to have 1
		 * document fall into few labels at once, with different "weight" for
		 * each.
		 */
		//构建了求质心的类 
		//通过获取WordVectors实现类WordVectorsImpl中的getLookupTable方法获取查询table及tokenizerFactory构造MeansBuilder类
		MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(),
				tokenizerFactory);
		
		//同理通过获取WordVectors实现类WordVectorsImpl中的getLookupTable方法获取查询table及标签列表构造LabelSeeker类    
		LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
				(InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

		int all=0;
		int correctCount=0;
		//遍历未分类文档  
		while (unClassifiedIterator.hasNextDocument()) {
			all++;
			LabelledDocument document = unClassifiedIterator.nextDocument();
			//把文档转成向量 
			INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
			//获取文档的类别得分  
			List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

			/*
			 * please note, document.getLabel() is used just to show which
			 * document we're looking at now, as a substitute for printing out
			 * the whole document name. So, labels on these two documents are
			 * used like titles, just to visualize our classification done
			 * properly
			 */
			log.info("Document '" + document.getLabels().get(0) + "' falls into the following categories: ");
			for (Pair<String, Double> score : scores) {
				log.info("        " + score.getFirst() + ": " + score.getSecond());
			}
			String preditcedLabel = "";
			preditcedLabel =  scores.get(0).getSecond() > scores.get(1).getSecond()?scores.get(0).getFirst():scores.get(1).getFirst();
			
			if(preditcedLabel.equals(document.getLabels().get(0))){
				correctCount++;
			}
		}

		System.out.println(correctCount);
		System.out.println(all);
		System.out.println(correctCount/(double)all);
	}
}
