Программная реализация статистического исследования для генератора xoshiro256**

Порядок запуска:

	1. 	g++ gen_table_256.cpp -o gen256
	   	./gen256 0
	   	создаётся dependency_table_xoshiro256_0.csv ; 2^24 строк ; init_s0,init_s1,init_s2,init_s3,result,next_s0,next_s1,next_s2,next_s3


	2.1. nvcc -arch=sm_86 collect_statistics256.cu -o collect_statistics256 --ptxas-options=-v
	     ./collect_statistics256
	     dependency_table_xoroshiro256_0.csv	->	./data/analysis_results256.csv
	     создаётся ./data/analysis_results256.csv ; 2^24 строк ;	Block,Row,Result,
	     														 	Correlation_s0,...,Correlation_s3,
	     														 	Uniformity,
	     														 	Bit_0,...,Bit_63,Pair_0,...,Pair_3,Triple_0,...,Triple_7,Quad_0,...,Quad_15,
	     														 	Entropy_0,...,Entropy_63,
	     														 	GlobalFrequency,RunsTest,BlockFrequency_0,...,BlockFrequency_15,
	     														 	MutualInformation_s0_s1,MutualInformation_s0_s2,MutualInformation_s0_s3,
																 	MutualInformation_s1_s2,MutualInformation_s1_s3,
																 	MutualInformation_s2_s3,
																 	MutualInformation_res_s0,...,MutualInformation_res_s3,
	     														 	Autocorr_0,...,Autocorr_9,
	     														 	Difference_0,...,Difference_255,
	     														 	Cluster_0,...,Cluster_31
	     														 	ClusterCount_0,...,ClusterCount_31

	2.2. nvcc -arch=sm_86 collect_statistics256_spectral_analysis.cu -o collect_statistics256_spectral_analysis --ptxas-options=-v -lcufft
		 ./collect_statistics256_spectral_analysis
		 dependency_table_xoroshiro256_0.csv	->	./data/analysis_results256_spectral.csv
		 создаётся ./data/analysis_results256_spectral.csv ; 2^24 строк ; Block,Row,
		 																  Spectral_0,Spectral_1,...,Spectral_511
	
	3.1. nvcc -arch=sm_86 pre_visualization_pocessing256.cu -o pre_visualization_pocessing256 --ptxas-options=-v
		 ./pre_visualization_pocessing256
		 ./data/analysis_results256.csv	->	./data/processed_data256.csv
		 создаётся ./data/processed_data256.csv ; 2^12 строк ;	Correlation_mean_0,Correlation_dispersion_0,...,Correlation_mean_3,Correlation_dispersion_3,
	     													   	Uniformity_mean_0,
	     													  	BitFrequency_mean_0,BitFrequency_entropy_0,...,BitFrequency_mean_63,BitFrequency_entropy_63,
	     													  	Pairs_mean_0,Pairs_dispersion_0,...,Pairs_mean_3,Pairs_dispersion_3,
	     													  	Triples_mean_0,Triples_dispersion_0,...,Triples_mean_7,Triples_dispersion_7,
	     													  	Quads_mean_0,Quads_dispersion_0,...,Quads_mean_15,Quads_dispersion_15,
	     													   	Entropy_mean_0,Entropy_stDev_0,...,Entropy_mean_63,Entropy_stDev_63,
			 													GlobalStats_mean_0,GlobalStats_dispersion_0,
			 													RunsTest_mean_0,RunsTest_dispersion_0,
			 													BlockFrequency_mean_0,...,BlockFrequency_dispersion_15,
	     													   	MutualInformation_mean_0,MutualInformation_dispersion_0,...,MutualInformation_mean_9,MutualInformation_dispersion_9,
	     													   	Autocorrelation_mean_0,Autocorrelation_stDev_0,...,Autocorrelation_mean_9,Autocorrelation_stDev_9,
	     													   	Differences_mean_0,...,Differences_mean_255,
	     													   	Clusters_mean_0,Clusters_distance_0_1,...,Clusters_mean_30,Clusters_distance_30_31,Clusters_mean_31,Clusters_distance_31_0,
	     													   	ClusterCount_mean_0,...,ClusterCount_mean_31
	     													  
	3.2. nvcc -arch=sm_86 pre_visualization_spectral_analysis256.cu -o pre_visualization_spectral_analysis256 --ptxas-options=-v
		 ./pre_visualization_spectral_analysis256
		 ./data/analysis_results256_spectral.csv	->	./data/processed_spectral_analysis256.csv
		 создаётся ./data/processed_spectral_analysis256.csv; 2^16 строк ; Block,Spectral_0,Spectral_1,...,Spectral_511


	4.1. python3 graphics_analysis256.py
		 ./data/processed_data256.csv -> ./picture256/picture.png x15
		 создаётся ./picture256/picture.png x15
	
	4.2. python3 graphics_analysis_spectral256.py
		 ./data/processed_spectral_analysis256.csv	->	./picture256/averaged_analysis_spectral256.png
		 создаётся ./picture256/averaged_analysis_spectral256.png  ; рисунок ; Спектральный анализ 256
