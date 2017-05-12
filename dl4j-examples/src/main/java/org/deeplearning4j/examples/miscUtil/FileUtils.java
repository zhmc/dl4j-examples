package org.deeplearning4j.examples.miscUtil;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class FileUtils {

	public static String readFileByLine(String string, String encoding)  {
		StringBuilder content = new StringBuilder();
		FileInputStream fr = null;
		BufferedReader br = null;

		try {
			fr = new FileInputStream(string);
			br = new BufferedReader(new InputStreamReader(fr, encoding));
			String line = "";
			while ((line = br.readLine()) != null) {
				content .append(  line+"\n");
			}
			br.close();
			fr.close();
		}catch (IOException e) {
			e.printStackTrace();
		}
		if(content.length()>0){
		content.deleteCharAt(content.length()-1);}
		return content.toString();
	}

	public static boolean isNumeric(String str) {
		Pattern pattern = Pattern.compile("^\\d+\\.?\\d+$|^\\d+$");
		Matcher isNum = pattern.matcher(str);
		if (!isNum.matches()) {
			return false;
		}
		return true;
	}

	/**
	 * 从本地读取对象
	 * @author zhminchao@163.com
	 * @date 2017年3月9日 下午1:40:39
	 * @param filepath
	 * @param toWhatClass 对象的类信息
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static <T> T readLocalObject(String filepath, Class<T> toWhatClass) {
		T obj=null;
		FileInputStream freader;
		try {
			freader = new FileInputStream(filepath);
			ObjectInputStream objectInputStream = new ObjectInputStream(freader);
			obj = (T) objectInputStream.readObject();
			
			objectInputStream.close();
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (ClassNotFoundException e2) {
			e2.printStackTrace();
		} catch ( IOException e) {
			e.printStackTrace();
		}
		
		return obj;
	}
	
	/**
	 * 将对象写到本地
	 * @author zhminchao@163.com
	 * @date 2017年3月9日 下午1:40:27
	 * @param obj
	 * @param filepath
	 */
	public static void writeLocalObject(Object obj, String filepath) {
		try {
			FileOutputStream outStream = new FileOutputStream(filepath);

			ObjectOutputStream object_OutputStream = new ObjectOutputStream(outStream);
			object_OutputStream.writeObject(obj);
			outStream.close();
			System.out.println("object has been written successfully");
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		} catch (IOException e2) {
			e2.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		// readFileByLine(new
		// File("C:\\Users\\Administrator\\Desktop\\13.txt"),"gbk");
		System.out.println(readFileByLine("slogs/slog.txt", "utf-8"));
	}
}
