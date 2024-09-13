import { observer } from "mobx-react-lite";
import langService from "../service/langService";
import img from "../assets/new/swap.svg";
import Button from "./Button";
import { useState } from "react";

import styles from "./styles/buttons/lang.module.scss";

const LangButt = observer(() => {
    const [isActive, setIsActive] = useState(false);

    const dynamicStyle = {
        transform: isActive ? 'rotateY(180deg)' : 'rotateY(0deg)',
        
    };

    const tap = ()=>{
        langService.toggleLang();
        setIsActive(!isActive);
    }    
    return(
        <div className={styles.langButt} style={dynamicStyle}>
            <Button img={img} tap={tap}  alt={"<->"}/>
        </div>
    );
});

export default LangButt;