const dn = require('dannjs');
const Dann = dn.dann;

const nn = new Dann(4,4);
nn.addHiddenLayer(16,'sigmoid');
nn.log();
nn.addHiddenLayer(16,'sigmoid');

nn.makeWeights();
nn.log({details: true, table: true});
