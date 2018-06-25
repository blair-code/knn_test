extern crate rustlearn;
extern crate rusty_machine;

mod data_loader;
mod knn_shapley;

use rusty_machine::prelude::{BaseMatrix, BaseMatrixMut, Vector, Matrix};
use data_loader::data_loader::{CSVLoader, DataLoader};
use knn_shapley::knn_shapley::*;
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

    let num_splits = 5;
    let test_percentage = 0.2;

    let mut split_shapleys = Matrix::zeros(num_splits, X.rows());

    println!("{:?}", split_shapleys.get_row(0));
    //for (train_idx, test_idx) in ShuffleSplit::new(X.rows(), num_splits, test_percentage) {
    for (split_index, (train_idx, test_idx)) in CrossValidation::new(X.rows(), num_splits).enumerate() {
        let X_train = X.select_rows(&train_idx);
        let y_train = y.select(&train_idx);
        let X_test = X.select_rows(&test_idx);
        let y_test = y.select(&test_idx);
        
        let shapleys = calculate_knn_shapleys(&X_train, &y_train, &X_test, &y_test, 10).into_vec();
        
        let mut row_slice = split_shapleys.get_row_mut(split_index).unwrap();
        for (i, item) in train_idx.iter().enumerate() {
            row_slice[*item] = shapleys[i];
        }
        // TODO Use indicator matrix to track which shapley values are set and which are just the default 0
        // TODO Average over columns to get shapley value of each sample (avg only over indicated values)

    }
    let cv_shapleys: Vec<f64> = split_shapleys.sum_rows().iter().map(|shapley_sum| shapley_sum/((num_splits-1) as f64)).collect(); 
    println!("{:?}", cv_shapleys);
    //println!("{:?}", split_shapleys.get_row(0));
}
