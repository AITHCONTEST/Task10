
import styles from "./styles/button.module.scss"
import keyboard from "../assets/keyboardNutt.svg";
import { useState } from "react";
import Keyboard from "./Keyboard";


interface KeyboardButtProps{
    get: () => string;
    set: (input:string) => void;
}

const KeyboardButt = ({get, set}:KeyboardButtProps) => {
    const [isKeyboard, setIsKeyboard] = useState<boolean>(false);

    const tap = () => {
        setIsKeyboard(!isKeyboard)
        const footer = document.querySelector("body");
        if (footer) {
            footer.style.paddingBottom = isKeyboard ? "20px" : "210px";
        }
    }
    
    return(
        <>
            <div onClick={tap} className={styles.butt}>
                <img src={keyboard} alt="keyboard" />
            </div>
            {isKeyboard && <Keyboard get={get} set={set} esc={tap}/>}
        </>
        
    );
};

export default KeyboardButt;