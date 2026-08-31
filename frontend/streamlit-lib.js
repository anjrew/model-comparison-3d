/**
 * Minimal Streamlit component client for the webgl_check component.
 *
 * Implements only the subset of the Streamlit component API used by this
 * component: setComponentReady, setComponentValue, and setFrameHeight.
 * See https://docs.streamlit.io/develop/concepts/custom-components
 */
var Streamlit = (function () {
    var ComponentMessageType = {
        COMPONENT_READY: "streamlit:componentReady",
        SET_COMPONENT_VALUE: "streamlit:setComponentValue",
        SET_FRAME_HEIGHT: "streamlit:setFrameHeight",
    };

    var registeredMessageListener = false;

    function sendBackMsg(type, data) {
        window.parent.postMessage(
            Object.assign({ isStreamlitMessage: true, type: type }, data),
            "*"
        );
    }

    return {
        API_VERSION: 1,

        setComponentReady: function () {
            if (!registeredMessageListener) {
                window.addEventListener("message", onMessageEvent);
                registeredMessageListener = true;
            }
            sendBackMsg(ComponentMessageType.COMPONENT_READY, {
                apiVersion: Streamlit.API_VERSION,
            });
        },

        setComponentValue: function (value) {
            sendBackMsg(ComponentMessageType.SET_COMPONENT_VALUE, {
                value: value,
                dataType: "json",
            });
        },

        setFrameHeight: function (height) {
            sendBackMsg(ComponentMessageType.SET_FRAME_HEIGHT, { height: height });
        },
    };

    function onMessageEvent(event) {
        var data = event.data;
        if (!data || data.type !== "streamlit:render") {
            return;
        }
    }
})();
