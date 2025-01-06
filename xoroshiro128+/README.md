Программная реализация статистического исследования для генератора xoroshiro128+

Порядок запуска:

	1. 	g++ gen_table_128.cpp -o gen128
	   	./gen128 0
	   	random	->	dependency_table_xoroshiro128_0.csv
	   	создаётся dependency_table_xoroshiro128_0.csv ; 2^24 строк ; init_s0,init_s1,result,next_s0,next_s1 
	   
	   
	2.1. nvcc -arch=sm_86 collect_statistics128.cu -o collect_statistics128 --ptxas-options=-v
	     ./collect_statistics128
	     dependency_table_xoroshiro128_0.csv	->	./data/analysis_results128.csv
	     создаётся ./data/analysis_results128.csv ; 2^24 строк ; Block,Row,Result,
	     														 Correlation_s0,Correlation_s1,
	     														 Uniformity,
	     														 Bit_0,...,Bit_63,Pair_0,...,Pair_3,Triple_0,...,Triple_7,Quad_0,...,Quad_15,
	     														 Entropy_0,...,Entropy_63,
	     														 GlobalFrequency,RunsTest,BlockFrequency_0,...,BlockFrequency_15,
	     														 MutualInformation_s0_s1,MutualInformation_s0_res,MutualInformation_s1_res,
	     														 Autocorr_0,...,Autocorr_9,
	     														 Difference_0,...,Difference_255,
	     														 Cluster_0,...,Cluster_9
	     														 ClusterCount_0,...,ClusterCount_9

	2.2. nvcc -arch=sm_86 collect_statistics128_spectral_analysis.cu -o collect_statistics128_spectral_analysis --ptxas-options=-v -lcufft
		 ./collect_statistics128_spectral_analysis
		 dependency_table_xoroshiro128_0.csv	->	./data/analysis_results128_spectral.csv
		 создаётся ./data/analysis_results128_spectral.csv ; 2^24 строк ; Block,Row,
		 																  Spectral_0,Spectral_1,...,Spectral_511
	
	3.1. nvcc -arch=sm_86 pre_visualization_pocessing128.cu -o pre_visualization_pocessing128 --ptxas-options=-v
		 ./pre_visualization_pocessing128
		 ./data/analysis_results128.csv	->	./data/processed_data128.csv
		 создаётся ./data/processed_data128.csv ; 2^12 строк ; Correlation_mean_0,Correlation_dispersion_0,Correlation_mean_1,Correlation_dispersion_1,
	     													   Uniformity_mean_0,
	     													  	 BitFrequency_mean_0,BitFrequency_entropy_0,...,BitFrequency_mean_63,BitFrequency_entropy_63,
	     													  	 Pairs_mean_0,Pairs_dispersion_0,...,Pairs_mean_3,Pairs_dispersion_3,
	     													  	 Triples_mean_0,Triples_dispersion_0,...,Triples_mean_7,Triples_dispersion_7,
	     													  	 Quads_mean_0,Quads_dispersion_0,...,Quads_mean_15,Quads_dispersion_15,
	     													   Entropy_mean_0,Entropy_stDev_0,...,Entropy_mean_63,Entropy_stDev_63,
			 													 GlobalStats_mean_0,GlobalStats_dispersion_0,
			 													 RunsTest_mean_0,RunsTest_dispersion_0,
			 													 BlockFrequency_mean_0,...,BlockFrequency_dispersion_15,
	     													   MutualInformation_mean_0,MutualInformation_dispersion_0,...,MutualInformation_mean_2,MutualInformation_dispersion_2,
	     													   Autocorrelation_mean_0,Autocorrelation_stDev_0,...,Autocorrelation_mean_9,Autocorrelation_stDev_9,
	     													   Differences_mean_0,...,Differences_mean_255,
	     													   Clusters_mean_0,Clusters_distance_0_1,...,Clusters_mean_8,Clusters_distance_8_9,Clusters_mean_9,Clusters_distance_9_0,
	     													   ClusterCount_mean_0,...,ClusterCount_mean_9
	     													  
	3.2. nvcc -arch=sm_86 pre_visualization_spectral_analysis128.cu -o pre_visualization_spectral_analysis128 --ptxas-options=-v
		 ./pre_visualization_spectral_analysis128
		 ./data/analysis_results128_spectral.csv	->	./data/processed_spectral_analysis128.csv
		 создаётся ./data/processed_spectral_analysis128.csv; 2^16 строк ; Block,Spectral_0,Spectral_1,...,Spectral_511


	4.1. python3 graphics_analysis128.py
		 ./data/processed_data128.csv -> ./picture128/picture.png x15
		 создаётся ./picture128/picture.png x15
	
	4.2. python3 graphics_analysis_spectral128.py
		 ./data/processed_spectral_analysis128.csv	->	./picture128/averaged_analysis_spectral128.png
		 создаётся ./picture128/averaged_analysis_spectral128.png  ; рисунок ; Спектральный анализ 128
