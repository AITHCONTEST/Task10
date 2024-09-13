import keyboard from "../assets/new/keyboard.svg"
import { useState } from "react";
import Keyboard from "./Keyboard";
import Button from "./Button";
import langStore from "../store/langStore";


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
            <div className={`${langStore.fromLang}`} style={{height:"100%"}}>
               <Button img={keyboard} tap={tap}  alt={"keyboard"}/> 
            </div>
            
            {isKeyboard && <Keyboard get={get} set={set} esc={tap}/>}
        </>
        
    );
};

export default KeyboardButt;