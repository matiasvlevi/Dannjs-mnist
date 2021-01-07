const fs = require('fs');

const mnist = require('mnist-data');
const test = mnist.testing(0,10000);
const train = mnist.training(0,60000);

let trainlabels = [];
let testlabels = [];
let trainImgs = [];
let testImgs = [];
function map(x,a,b,c,d) {
    return (x-a)/(b-a)* (d-c) + c;
}
function format(arr) {
    let newArr = [];
    for (let i = 0; i < arr.length; i++) {
        newArr[i] = map(arr[i],0,255,0,1);
    }
    return newArr;
}
function merge(arr) {
    let array = [];
    for (let i = 0; i < arr.length; i++) {
        let o = array.concat(format(arr[i]));
        array = o;
    }
    return array;
}
function makeLabel(x,l) {
    let arr = [];
    for (let i = 0; i < l; i++) {
        if (i == x) {
            arr[i] = 1;
        } else {
            arr[i] = 0;
        }
    }
    return arr;
}

for (let i = 0; i < 60000; i++) {
    trainlabels[i] = makeLabel(train.labels.values[i],10);
    trainImgs[i] = merge(train.images.values[i]);
}
for (let i = 0; i < 10000; i++) {
    testlabels[i] = makeLabel(test.labels.values[i],10);
    testImgs[i] = merge(test.images.values[i]);
}

console.log("Test data: "+test.labels.values.length);
console.log("Train data: "+train.images.values.length);

let dannjs = require('dannjs');
let Dann = dannjs.dann;
let Layer = dannjs.layer;
let Matrix = dannjs.matrix;
let activations = dannjs.activations;
let pickFuncs = dannjs.pickFuncs;

let name = 'samplec_v2_1_9';
//Creating a pooling layer to downsample (784 to 196):
let dsl = new Layer('avgpool',784,2,2);
//Creating the Deep Neural Network:
let nn = new Dann(196,11);
// nn.addHiddenLayer(121,'leakyReLU');
// nn.addHiddenLayer(81,'leakyReLU');
// nn.addHiddenLayer(49,'leakyReLU');
// nn.setLossFunction('bce');
// nn.makeWeights(-0.1,0.1);
nn.load('sampleB_v2_1_9');

nn.lr = 0.000001;
nn.log();

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
function makeEmpty(num) {
    let arr = [];
    for (let i = 0; i < num; i++) {
        arr[i] = 0;
    }
    return arr;
}

//train & test :
function train_(batchPerEpoch,epoch) {
    let len = train.labels.values.length;
    for(let e = 0; e < epoch; e++) {
        console.log("Start Epoch: "+nn.epoch);
        let sum = 0;
        let batchlength = len/batchPerEpoch;
        for (let j=0; j < batchPerEpoch; j++) {
            let bsum = 0;
            for (let k = 0; k < batchlength; k++) {
                let index = (j*batchlength)+k;
                let input = trainImgs[index];
                let downsampled = dsl.feed(input);
                let target = trainlabels[index];
                target[10] = 0;
                //console.log(downsampled.length,nn.i)
                nn.backpropagate(downsampled,target);

                bsum += nn.loss;
            }
            for (let i = 0; i < batchlength/10; i++) {

                let downsampled = makeEmpty(196);
                let target = [0,0,0,0,0,0,0,0,0,0,1];
                nn.backpropagate(downsampled,target);

            }
            //nn.losses.push(bsum);
            sum += bsum;
            let bavgLoss = bsum/batchlength;
            console.log("Batch: "+j+" AvgLoss: "+bavgLoss);
        }

        let avgloss = sum/len;
        let result = test_();
        console.log("  ");
        console.log("Completed Epoch: "+nn.epoch+ "  Accuracy: "+result+" AvgLoss: "+avgloss);
        console.log("  ");
        nn.epoch++;
    }

}
function test_() {
    let points = 0;
    let len = testlabels.length;
    for (let i = 0; i < len; i++) {
        let downsampled = dsl.feed(testImgs[i]);
        let target = testlabels[i];
        target[10] = 0;
        let out = nn.feedForward(downsampled,target);
        let predictedLabel = findLargest(out);
        let realLabel = test.labels.values[i];

        if (predictedLabel == realLabel) {
            points++;
        }
    }
    return points/len;
}

for (let i = 0; i < 100; i++) {
    train_(100,1);
    nn.save(name,test_);
}
