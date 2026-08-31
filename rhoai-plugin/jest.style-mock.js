// Simple CSS module mock - returns the class name as-is
module.exports = new Proxy({}, {
  get: function(target, prop) {
    return prop;
  }
});
