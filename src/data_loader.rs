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
        fn load_all_samples_from_disk(&self, filepaths: Vec<&str>) -> (Matrix<Self::Feature>,Vector<Self::Predictor>);
    }

    pub struct CSVLoader {
        pub delimiter: u8,
        pub predictor_column_index: usize,
        pub has_headers: bool
    }

    impl DataLoader for CSVLoader{
        type Feature = f64;
        type Predictor = u32;
        fn load_all_samples_from_disk(&self, filepaths: Vec<&str>) -> (Matrix<Self::Feature>,Vector<Self::Predictor>) {
            
            let mut all_features = Vec::new();
            let mut all_predictors = Vec::new();
            let mut samplecount = 0;
            let mut featurecount = None;
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
                    match featurecount {
                        None => featurecount = Some(rr.len() - 1),
                        Some(_) => (),
                    };
                    for i in 0..rr.len() {
                        if i != self.predictor_column_index {
                            rec_vec.push(rr.get(i).unwrap().parse::<Self::Feature>().unwrap());
                        } else {
                            all_predictors.push(rr.get(i).unwrap().parse::<Self::Predictor>().unwrap());
                        }
                    }
                    samplecount += 1;
                    all_features.push(rec_vec.to_owned());
                }
            }
            
            println!("samplecount {} featurecount {}", samplecount, featurecount.unwrap());
            let flattened_features = all_features.iter().flat_map(|x| x.to_owned()).collect();
            let feature_matrix = Matrix::new::<Vec<f64>>(samplecount, featurecount.unwrap(),flattened_features);
            let predictor_vector = Vector::new(all_predictors);
            //let predictor_vector = Vector::new(vec![0; 5]);
            (feature_matrix, predictor_vector)
        }
    }
}