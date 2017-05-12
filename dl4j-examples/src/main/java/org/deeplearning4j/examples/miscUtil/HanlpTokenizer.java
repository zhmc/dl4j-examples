package org.deeplearning4j.examples.miscUtil;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import com.hankcs.hanlp.seg.common.Term;
import com.hankcs.hanlp.tokenizer.NotionalTokenizer;

/**
 * 包装了Hanlp的中文分词器，用来代替默认的空格分词器。<br>
 * 但是实际上跑起来的时候，有一定概率会发生字典文件读取失败的事情<br>
 * <p>初步推测应该是由于多线程同时读取时导致出错</p>
 * <p>替代方案时是用一个预处理的方法，先将原始文档转换为空格切分的文档，然后就用默认的空格分词器</p>
 * @author zhminchao@163.com
 * @date 2017年5月10日 下午4:56:56
 */
public class HanlpTokenizer implements Tokenizer {

	private String text;

	public HanlpTokenizer(String text) {
		this.text=text;
	}
	@Override
	public boolean hasMoreTokens() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public int countTokens() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public String nextToken() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<String> getTokens() {
		List<String> wordList = new ArrayList<String>();
		List<Term> termList = NotionalTokenizer.segment(text);

//		words = new String[termList.size()];
		for (int i = 0; i < termList.size(); i++) {
			//去除人名
			if(termList.get(i).toString().contains("nr")){
				continue;
			}
			
			wordList.add(termList.get(i).toString().replace("\n", ""));
		}
		return wordList;
	}

	@Override
	public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
		// TODO Auto-generated method stub
		
	}

}
