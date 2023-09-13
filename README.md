# The Fork
This is a custom modified fork of the Google TensorFlow models that I use for my performance [\(d\)instances](https://www.samuelealbani.com/works/dinstances).

## Development

You can run the unit tests for any of the models by running the following
inside a directory:

`yarn test`

New models should have a test NPM script (see [this](./mobilenet/package.json) `package.json` and `run_tests.ts` [helper](./mobilenet/run_tests.ts) for reference).

To run all of the tests, you can run the following command from the root of this
repo:

`yarn presubmit`
