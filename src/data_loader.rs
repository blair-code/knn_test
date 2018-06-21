extern crate rusty_machine;
extern crate rulinalg;
extern crate csv;

pub mod data_loader {

    use data_loader::rusty_machine::linalg::Matrix;
    use data_loader::rusty_machine::linalg::Vector;
    use data_loader::csv;

    pub trait DataLoader {
        type Feature;
        type Predictor;
        fn load_all_samples(&self, filepaths: Vec<&str>) -> (Vec<Vec<Self::Feature>>, Vec<Self::Predictor>);
        fn vecs_as_matrix(&self, features: Vec<Vec<Self::Feature>>) -> Matrix<Self::Feature> {
            let samplecount = features.len();
            let featurescount = if samplecount > 0 {features[0].len()} else {0};
            let flattened_features = features.into_iter().flat_map(|x| x).collect();
            Matrix::new::<Vec<Self::Feature>>(samplecount, featurescount, flattened_features)
        }

        fn vec_as_vector(&self, predictors: Vec<Self::Predictor>) -> Vector<Self::Predictor> {
            Vector::new(predictors)
        }
    }

    pub struct CSVLoader {
        pub delimiter: u8,
        pub predictor_column_index: usize,
        pub has_headers: bool
    }

    impl DataLoader for CSVLoader{
        type Feature = f64;
        type Predictor = u32;
        fn load_all_samples(&self, filepaths: Vec<&str>) -> (Vec<Vec<Self::Feature>>, Vec<Self::Predictor>) {
            
            let mut all_features = Vec::new();
            let mut all_predictors = Vec::new();

            for path in filepaths.iter() {
                let mut rdr = csv::ReaderBuilder::new()
                                .has_headers(self.has_headers)
                                .delimiter(self.delimiter)
                                .flexible(false)
                                .from_path(path)
                                .unwrap();
                
                
                for rec in rdr.records() {
                    let rr = rec.unwrap();
                    let mut rec_vec = Vec::new();

                    for i in 0..rr.len() {
                        if i != self.predictor_column_index {
                            rec_vec.push(rr.get(i).unwrap().parse::<Self::Feature>().unwrap());
                        } else {
                            all_predictors.push(rr.get(i).unwrap().parse::<Self::Predictor>().unwrap());
                        }
                    }
                    all_features.push(rec_vec.to_owned());
                }
            }

            (all_features, all_predictors)
        }
    }
}