package com.said.example.Kmeans;


import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.time.Duration;
import java.time.Instant;
import scala.util.Random;

import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Pattern;
import java.util.Arrays;
import java.util.List;
import java.lang.Iterable;
import scala.Tuple2;
import org.apache.commons.lang.StringUtils;
import org.apache.commons.collections.IteratorUtils;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.util.Vector;

import scala.Tuple2;

public final class Kmeans {
	private static final Pattern COMMA = Pattern.compile(",");
	private static JavaSparkContext context;

	public static void main(String[] args) throws Exception {

		String path = args[0];
		int k = Integer.parseInt(args[2]);
		int maxIterations = Integer.MAX_VALUE;
		double convergenceEpslon = 0; // default
		if (args.length > 3) {
			if (!args[3].equals("MAX")) { 
				maxIterations = Integer.parseInt(args[3]);
			}
		}
		if (args.length > 4) {
			convergenceEpslon = Double.parseDouble(args[4]);
		}

		SparkConf conf = new SparkConf().setMaster("local").setAppName("kmeans");
		context = new JavaSparkContext(conf);

		JavaRDD<Vector> data = context.textFile(path)
				.map(new Function<String, Vector>() {
					@Override
					public Vector call(String line) {
						return parseVector(line);
					}
				}).cache();

		context.parallelize(kmeans(data, k, convergenceEpslon, maxIterations)).saveAsTextFile(args[1]);


		System.exit(0);
	}

	public static List<Vector> kmeans(JavaRDD<Vector> data, int k,
			double convergeDist, long maxIterations) {

		final List<Vector> centroids = data.takeSample(false, k);
		long counter = 0;
		double tempDist;
		Instant start = Instant.now();
		do {
			
			JavaPairRDD<Integer, Vector> closest = data
					.mapToPair(new PairFunction<Vector, Integer, Vector>() {
						@Override
						public Tuple2<Integer, Vector> call(Vector vector) {
							return new Tuple2<Integer, Vector>(closestPoint(
									vector, centroids), vector);
						}
					});

			JavaPairRDD<Integer, Iterable<Vector>> pointsGroup = closest.groupByKey();
			Map<Integer, Vector> newCentroids = pointsGroup.mapValues(
					new Function<Iterable<Vector>, Vector>() {
						@Override
						public Vector call(Iterable<Vector> ps) {
							ArrayList<Vector> list = new ArrayList<Vector>();
    					if(ps != null) {
      					for(Vector e: ps) {
        						list.add(e);
      					}
    					}
							return average(list);
						}
					}).collectAsMap();
			tempDist = 0.0;
			for (int j = 0; j < k; j++) {
				tempDist += centroids.get(j).squaredDist(newCentroids.get(j));
			}
			for (Map.Entry<Integer, Vector> t : newCentroids.entrySet()) {
				centroids.set(t.getKey(), t.getValue());
			}
			counter++;
		} while (tempDist > convergeDist && counter < maxIterations);
		Instant end = Instant.now();
		Duration timeElapsed = Duration.between(start, end);
		System.out.println("Time taken: " + timeElapsed.toMillis() +" milliseconds");
		System.out.println("Converged in " + String.valueOf(counter) + " iterations.");
		System.out.println("Final centers:");
		for (Vector c : centroids) {
			System.out.println(c);
		}
		return centroids;
	}

	static Vector parseVector(String line) {
		String[] splits = COMMA.split(line);
		double[] data = new double[splits.length];
		int i = 0;
		for (String s : splits) {
			data[i] = Double.parseDouble(s);
			i++;
		}
		return new Vector(data);
	}

	static int closestPoint(Vector p, List<Vector> centers) {
		int bestIndex = 0;
		double closest = Double.POSITIVE_INFINITY;
		for (int i = 0; i < centers.size(); i++) {
			double tempDist = p.squaredDist(centers.get(i));
			if (tempDist < closest) {
				closest = tempDist;
				bestIndex = i;
			}
		}
		return bestIndex;
	}

	static Vector average(List<Vector> ps) {
		int numVectors = ps.size();
		Vector out = new Vector(ps.get(0).elements());
		for (int i = 1; i < numVectors; i++) {
			out.addInPlace(ps.get(i));
		}
		return out.divide(numVectors);
	}
}
