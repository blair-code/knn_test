extern crate rustlearn;
extern crate rusty_machine;

mod data_loader;
mod knn_shapley;

use rusty_machine::prelude::BaseMatrix;
use data_loader::data_loader::{CSVLoader, DataLoader};
use knn_shapley::knn_shapley::calculate_knn_shapleys;
use rustlearn::cross_validation::{ShuffleSplit, CrossValidation};

fn main() {
    let path = "./iris.data.transf";
    let csv_ldr = CSVLoader{
        delimiter: b',',
        predictor_column_index: 4,
        has_headers: false
    };

    let (features, predictors) = csv_ldr.load_all_samples(vec![path]);
    let X = csv_ldr.vecs_as_matrix(features);
    let y = csv_ldr.vec_as_vector(predictors);

    let num_splits = 10;
    let test_percentage = 0.2;

    //for (train_idx, test_idx) in ShuffleSplit::new(X.rows(), num_splits, test_percentage) {
    for (train_idx, test_idx) in CrossValidation::new(X.rows(), num_splits) {
        let X_train = X.select_rows(&train_idx);
        let y_train = y.select(&train_idx);
        let X_test = X.select_rows(&test_idx);
        let y_test = y.select(&test_idx);
        
        let shapleys = calculate_knn_shapleys(&X_train, &y_train, &X_test, &y_test, 3);
        println!("{:?}", shapleys);
    }
}
