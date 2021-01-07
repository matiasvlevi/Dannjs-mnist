var _ = require('underscore');
var fs = require('fs');
var path = require('path');

function get_images (fp, n, m) {
  // get data
  var buffer = new Buffer(16 + (Math.pow(28, 2) * (m)));
  var fd = fs.openSync(fp, 'r');
  fs.readSync(fd, buffer, 0, 16, 0);
  fs.readSync(fd, buffer, 16, (Math.pow(28, 2) * (m - n)), 16 + Math.pow(28, 2) * n);
  fs.closeSync(fd);
  // parse data
  var magic_number = buffer.readUInt32BE(0);
  var num_items = buffer.readUInt32BE(4);
  var num_rows = buffer.readUInt32BE(8);
  var num_cols = buffer.readUInt32BE(12);
  var images = _.range(m - n).map(function (i) {
    var offset = 16 + Math.pow(28, 2) * i;
    return _.range(28).map(function (j) {
      return _.range(28).map(function (k) {
        return buffer.readUInt8(offset + (28 * j) + k);
      });
    });
  });
  
  return {
    magic_number: magic_number,
    total_num_items: num_items,
    rows: num_rows,
    cols: num_cols,
    values: images,
    start: n,
    end: m
  };
}

function get_labels (fp, n, m) {
  // get data
  var buffer = new Buffer(8 + m);
  var fd = fs.openSync(fp, 'r');
  fs.readSync(fd, buffer, 0, 8, 0);
  fs.readSync(fd, buffer, 8, (m - n), 8 + n);
  fs.closeSync(fd);
  // parse data
  var magic_number = buffer.readUInt32BE(0);
  var num_items = buffer.readUInt32BE(4);
  var labels = _.range(m - n).map(function (i) {
    return buffer.readUInt8(8 + i);
  });
  
  return {
    magic_number: magic_number,
    total_num_items: num_items,
    values: labels,
    start: n,
    end: m
  };
}

function get_training_images (n, m) {
  var fp = path.join(__dirname, 'data', 'train-images-idx3-ubyte');
  return get_images(fp, n, m);  
}

function get_training_labels (n, m) {
  var fp = path.join(__dirname, 'data', 'train-labels-idx1-ubyte');
  return get_labels(fp, n, m);
}

function get_testing_images (n, m) {
  var fp = path.join(__dirname, 'data', 't10k-images-idx3-ubyte');
  return get_images(fp, n, m);
}

function get_testing_labels (n, m) {
  var fp = path.join(__dirname, 'data', 't10k-labels-idx1-ubyte');
  return get_labels(fp, n, m);
}

/**
@module MNIST
@description Utilities for working with the MNIST dataset of handwritten images.
@example
// require the mnist-data module
var mnist = require('mnist-data');
@example
// retrieve a slice of the first 100 objects from the training set
var training_data = mnist.training(100);
training_data.labels.values[0]; // the label 0-9 of the first handwritten digit in the slice
training_data.images.values[0]; // the matrix of 0-255 greyscale values for each pixel of the 28x28 image
@example
// retrieve a slice of the first 100 objects from the testing set
var testing_data = mnist.testing(100);
// the label 0-9 of the first handwritten digit in the slice
testing_data.labels.values[0];
// the matrix of 0-255 greyscale values for each pixel of the 28x28 image
testing_data.images.values[0];
*/
module.exports = {
  /**
  @description Retrieve a slice (n,m) from the `train-images-idx3-ubyte` and `train-labels-idx1-ubyte` files. 
  @function
  @param {number} n - start index of training labels and images to retrieve. if m is undefined, n becomes the end index, while the start index becomes 0.
  @param {number} m - (optional) end index of training labels and images to retrieve.
  @returns {object} images: data about the images, labels: data about the labels
  */
  training: function (n, m) {
    if (m === undefined) {
      m = n;
      n = 0;
    }

    return {
      images: get_training_images(n, m),
      labels: get_training_labels(n, m)
    };
  },
  /**
  @description Retrieve a slice (n,m) from the `t10k-images-idx3-ubyte` and `t10k-labels-idx1-ubyte` files. 
  @function
  @param {number} n - start index of training labels and images to retrieve. if m is undefined, n becomes the end index, while the start index becomes 0.
  @param {number} m - (optional) end index of training labels and images to retrieve.
  @returns {object} images: data about the images, labels: data about the labels
  */
  testing: function (n, m) {
    if (m === undefined) {
      m = n;
      n = 0;
    }

    return {
      images: get_testing_images(n, m),
      labels: get_testing_labels(n, m)
    };
  }
};
