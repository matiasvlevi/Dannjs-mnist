const mnist = require('mnist');
const set = mnist.set(60000-100, 100);
const trainData = set.training;
const testData = set.test;

const dn = require('dannjs');
const Dann = dn.dann;

const nn = new Dann(784,10);
nn.addHiddenLayer(256,'leakyReLU');
nn.addHiddenLayer(128,'leakyReLU');
nn.addHiddenLayer(32,'leakyReLU');
nn.makeWeights();
nn.setLossFunction('bce');
nn.lr = 0.0001;
nn.log();

let batchNb = 100;
function findLargest(arr) {
    let record = 0;
    let bestIndex = 0;
    for (let i = 0; i < arr.length; i++) {
        if (arr[i] > record) {
            record = arr[i];
            bestIndex = i;
        }
    }
    return bestIndex;
}
function test() {
    let points = 0;
    for (data of testData) {
        let out = nn.feedForward(data.input,data.output);
        let predictedLabel = findLargest(out);
        let realLabel = findLargest(data.output);
        if (predictedLabel == realLabel) {
            points++;
        }
    }
    return points/testData.length;
}

for (let i = 0; i < 20; i++) {
    let sum = 0;

    let batchlength = trainData.length/batchNb;
    for (let j=0; j < batchNb; j++) {
        let bsum = 0;
        for (let k = 0; k < batchlength; k++) {
            let data = trainData[(j*batchlength)+k];

            nn.backpropagate(data.input,data.output);

            bsum += nn.loss;
        }
        sum += bsum;
        let bavgLoss = bsum/batchlength;
        console.log("Batch: "+j+" AvgLoss: "+bavgLoss);
    }
    nn.epoch++;
    let avgloss = sum/trainData.length;
    let result = test();
    console.log("  ")
    console.log("Epoch: "+nn.epoch+ "  Accuracy: "+result+" AvgLoss: "+avgloss);
}
nn.save('sample');
//
// nn.log({weights: true, table: false});
