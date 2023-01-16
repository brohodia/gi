/* eslint-disable */
import * as HX from "hx-model-components";

function HXModel() {
  return (
    // Your model code goes here
    <HX.Root>
      <HX.Page>
        <HX.Section title="My fields">
          <HX.Pane>
            <HX.Collection fields={["a", "b", "c"]} />
          </HX.Pane>
        </HX.Section>
      </HX.Page>
    </HX.Root>
  );
}

export default HXModel;
