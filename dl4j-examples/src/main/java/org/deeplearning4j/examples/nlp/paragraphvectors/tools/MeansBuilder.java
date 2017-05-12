package org.deeplearning4j.examples.nlp.paragraphvectors.tools;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Simple utility class that builds centroid vector for LabelledDocument
 * based on previously trained ParagraphVectors model
 *
 * @author raver119@gmail.com
 */
public class MeansBuilder {
    private VocabCache<VocabWord> vocabCache;                //词汇表
    private InMemoryLookupTable<VocabWord> lookupTable;      //查询table
    private TokenizerFactory tokenizerFactory;               //分词器

    //构造方法，根据传入的参数赋值当前对象的词汇表，查询table,分词器
    public MeansBuilder(@NonNull InMemoryLookupTable<VocabWord> lookupTable,
        @NonNull TokenizerFactory tokenizerFactory) {
        this.lookupTable = lookupTable;
        this.vocabCache = lookupTable.getVocab();
        this.tokenizerFactory = tokenizerFactory;
    }

    /**
     * This method returns centroid (mean vector) for document.
     *
     * @param document
     * @return
     */
    public INDArray documentAsVector(@NonNull LabelledDocument document) {
    	//切割文档，获取词列表
        List<String> documentAsTokens = tokenizerFactory.create(document.getContent()).getTokens();
        //声明一个原子整数0，保证线程安全
        AtomicInteger cnt = new AtomicInteger(0);
        //统计独立词计数
        for (String word: documentAsTokens) {
            if (vocabCache.containsWord(word)) cnt.incrementAndGet();
        }
        if(cnt.get()==0){
        	System.out.println("这篇文章就1个词");
        	INDArray a = Nd4j.create(1, lookupTable.layerSize());
        	return a;
        }
        //根据词计数构建词矩阵，行是词计数，列是每个词对应的向量长度，默认100
        INDArray allWords = Nd4j.create(cnt.get(), lookupTable.layerSize());

        cnt.set(0);//词计数清零
        //给词矩阵赋值，
        for (String word: documentAsTokens) {
            if (vocabCache.containsWord(word))
            	//根据词表索引，取出对应词权重向量的行，放入allWords矩阵
                allWords.putRow(cnt.getAndIncrement(), lookupTable.vector(word));
        }

        //通过mean(0)把矩阵合成一行，0代表维度，也是就求质心并返回
        INDArray mean = allWords.mean(0);

        return mean;
    }
}
